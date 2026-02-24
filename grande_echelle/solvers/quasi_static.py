import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc

try:
    from ..loads import moving_gaussian_pressure
except ImportError:  # pragma: no cover - script execution fallback
    from loads import moving_gaussian_pressure


def _write_monitor_csv(path, rows):
    lines = ["step,time,max_u_inf,max_damage,mean_damage,frac_damage_ge_095"]
    for step, time_value, max_u, max_d, mean_d, frac_d95 in rows:
        lines.append(
            f"{step},{time_value:.12g},{max_u:.12e},{max_d:.12e},{mean_d:.12e},{frac_d95:.12e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_phase_field_params(cfg, phase_field_preset):
    gc_value = cfg.phase_field_gc_j_m2
    l0_value = cfg.phase_field_l0_m
    if cfg.phase_field_use_selected_preset and phase_field_preset:
        gc_value = float(phase_field_preset.get("Gc_J_m2", gc_value))
        l0_value = float(phase_field_preset.get("l0_m", l0_value))
    return gc_value, l0_value


def run_quasi_static(model, cfg, output_layout, phase_field_preset=None):
    domain = model.domain
    t = fem.Constant(domain, 0.0)

    comm = domain.comm
    xmin = comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
    xmax = comm.allreduce(domain.geometry.x[:, 0].max(), op=MPI.MAX)
    ymin = comm.allreduce(domain.geometry.x[:, 1].min(), op=MPI.MIN)
    ymax = comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX)
    zmin = comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN)
    zmax = comm.allreduce(domain.geometry.x[:, 2].max(), op=MPI.MAX)

    iceberg_center_y = getattr(cfg, "iceberg_center_y", None)
    if iceberg_center_y is not None:
        y_mid = float(np.clip(iceberg_center_y, ymin, ymax))
    else:
        y_mid = cfg.y_mid_factor * ymax
    if hasattr(cfg, "waterline_z") and hasattr(cfg, "iceberg_depth_below_waterline"):
        z_target = cfg.waterline_z - cfg.iceberg_depth_below_waterline
        z_mid = float(np.clip(z_target, zmin, zmax))
    else:
        z_mid = zmin + cfg.z_mid_factor * (zmax - zmin)

    if getattr(cfg, "iceberg_moves_from_xmax_to_xmin", False):
        x0 = xmax
        x1 = xmin
    else:
        x0 = xmin
        x1 = xmax
    vx = (x1 - x0) / cfg.t_final

    c0 = fem.Constant(domain, (x0, y_mid, z_mid))
    v_ice = fem.Constant(domain, (vx, 0.0, 0.0))
    sigma = fem.Constant(domain, cfg.sigma)
    p0 = fem.Constant(domain, cfg.pressure_peak)

    # Default: no external Neumann load (used for Dirichlet-driven iceberg mode)
    zero_vec = fem.Constant(domain, (0.0, 0.0, 0.0))
    L = ufl.dot(zero_vec, model.u_test) * ufl.dx
    extra_bcs = []
    if cfg.iceberg_loading == "neumann_pressure":
        p = moving_gaussian_pressure(domain, c0, v_ice, t, sigma, p0)
        f_ice = -p * model.e3
        L = ufl.dot(f_ice, model.u_test) * ufl.dx
    elif cfg.iceberg_loading == "dirichlet_displacement":
        radius_y = cfg.iceberg_patch_radius_factor * cfg.sigma
        radius_z = cfg.iceberg_patch_radius_factor * cfg.sigma
        waterline_z = getattr(cfg, "waterline_z", np.inf)

        def impact_region(x):
            y_scaled = (x[1] - y_mid) / max(radius_y, 1e-12)
            z_scaled = (x[2] - z_mid) / max(radius_z, 1e-12)
            inside_patch = (y_scaled * y_scaled + z_scaled * z_scaled) <= 1.0
            submerged = x[2] <= waterline_z
            return inside_patch & submerged

        ice_dofs = fem.locate_dofs_geometrical((model.V.sub(0), model.Vu), impact_region)
        u_ice = fem.Function(model.Vu, name="IcebergDisplacement")
        extra_bcs.append(fem.dirichletbc(u_ice, ice_dofs, model.V.sub(0)))
    else:
        raise ValueError(
            f"Unknown cfg.iceberg_loading='{cfg.iceberg_loading}'. "
            "Expected 'neumann_pressure' or 'dirichlet_displacement'."
        )

    problem = dolfinx.fem.petsc.LinearProblem(
        model.a,
        L,
        u=model.v,
        bcs=model.bcs + extra_bcs,
        petsc_options=cfg.petsc_options,
        petsc_options_prefix="coque",
    )

    u_out = fem.Function(model.Vu, name="Displacement")
    theta_out = fem.Function(model.Vtheta, name="Rotation")
    Vd = model.Vd
    damage = model.damage_state
    damage.name = "Damage"
    damage_old = fem.Function(Vd, name="DamageOld")
    history = fem.Function(Vd, name="HistoryField")
    damage.x.array[:] = 0.0
    damage_old.x.array[:] = 0.0
    history.x.array[:] = 0.0

    damage_enabled = cfg.enable_global_phase_field
    if damage_enabled:
        gc_value, l0_value = _resolve_phase_field_params(cfg, phase_field_preset)
        gc = fem.Constant(domain, gc_value)
        l0 = fem.Constant(domain, l0_value)
        residual_stiffness = fem.Constant(domain, cfg.phase_field_residual_stiffness)

        u_sol, _ = ufl.split(model.v)
        P_plane = ufl.as_matrix(
            [
                [model.e1[0], model.e2[0]],
                [model.e1[1], model.e2[1]],
                [model.e1[2], model.e2[2]],
            ]
        )
        t_grad_u = ufl.dot(ufl.grad(u_sol), P_plane)
        eps = ufl.sym(ufl.dot(P_plane.T, t_grad_u))
        lmbda = model.E_field * model.nu_field / (1 + model.nu_field) / (1 - 2 * model.nu_field)
        mu = model.E_field / 2 / (1 + model.nu_field)
        lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
        psi_drive_expr = 0.5 * lmbda_ps * ufl.tr(eps) ** 2 + mu * ufl.inner(eps, eps)
        psi_drive = fem.Function(Vd, name="PsiDrive")
        psi_eval = fem.Expression(psi_drive_expr, Vd.element.interpolation_points)

        dd = ufl.TrialFunction(Vd)
        eta = ufl.TestFunction(Vd)
        a_d = (
            gc * l0 * ufl.dot(ufl.grad(dd), ufl.grad(eta))
            + (gc / l0 + 2.0 * history + residual_stiffness) * dd * eta
        ) * ufl.dx
        L_d = (2.0 * history) * eta * ufl.dx
        damage_problem = dolfinx.fem.petsc.LinearProblem(
            a_d,
            L_d,
            u=damage,
            bcs=[],
            petsc_options_prefix="coque_damage",
            petsc_options=cfg.petsc_options,
        )
    else:
        psi_drive = None
        psi_eval = None
        damage_problem = None

    time_steps = np.linspace(0.0, cfg.t_final, cfg.num_steps + 1)
    monitor_rows = []

    with io.VTKFile(MPI.COMM_WORLD, output_layout["displacement_file"], "w") as disp_vtk:
        with io.VTKFile(MPI.COMM_WORLD, output_layout["rotation_file"], "w") as rot_vtk:
            with io.VTKFile(MPI.COMM_WORLD, output_layout["damage_file"], "w") as damage_vtk:
                for n, tn in enumerate(time_steps):
                    t.value = tn
                    if cfg.iceberg_loading == "dirichlet_displacement":
                        x_center = x0 + vx * tn

                        def prescribed_displacement(x):
                            r2 = (x[0] - x_center) ** 2 + (x[2] - z_mid) ** 2
                            amplitude = cfg.iceberg_disp_sign * cfg.iceberg_disp_peak * np.exp(
                                -r2 / (2 * cfg.sigma**2)
                            )
                            return np.vstack(
                                (np.zeros_like(amplitude), amplitude, np.zeros_like(amplitude))
                            )

                        u_ice.interpolate(prescribed_displacement)
                    # Mechanical solve uses damage from the previous converged load step
                    # through `model.damage_state` embedded in the shell bilinear form.
                    problem.solve()
                    u_out.interpolate(model.v.sub(0))
                    theta_out.interpolate(model.v.sub(1))

                    if damage_enabled:
                        psi_drive.interpolate(psi_eval)
                        history.x.array[:] = np.maximum(
                            history.x.array, np.maximum(psi_drive.x.array, 0.0)
                        )
                        damage_problem.solve()
                        d_clipped = np.clip(damage.x.array, 0.0, 1.0)
                        d_irrev = np.maximum(damage_old.x.array, d_clipped)
                        damage.x.array[:] = d_irrev
                        damage_old.x.array[:] = d_irrev
                    else:
                        damage.x.array[:] = 0.0

                    disp_vtk.write_function(u_out, tn)
                    rot_vtk.write_function(theta_out, tn)
                    damage_vtk.write_function(damage, tn)

                    u_max = np.linalg.norm(u_out.x.array, ord=np.inf)
                    max_d = float(np.max(damage.x.array))
                    mean_d = float(np.mean(damage.x.array))
                    frac_d95 = float(np.mean(damage.x.array >= 0.95))
                    monitor_rows.append((n, tn, u_max, max_d, mean_d, frac_d95))
                    if MPI.COMM_WORLD.rank == 0:
                        print(
                            f"Step {n}/{len(time_steps)-1}, t={tn:.3e}, "
                            f"max|u|={u_max:.3e}, max(d)={max_d:.3e}, "
                            f"mean(d)={mean_d:.3e}, frac(d>=0.95)={frac_d95:.3e}"
                        )

    if MPI.COMM_WORLD.rank == 0:
        _write_monitor_csv(output_layout["monitor_file"], monitor_rows)
