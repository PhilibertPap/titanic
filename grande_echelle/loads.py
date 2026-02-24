import ufl
from dolfinx import fem

def gravity_load(domain, thick, rho=7850.0, g_vec=(0.0, 0.0, -9.81)):
    """
    Charge surfacique équivalente au poids propre pour une coque 2D
    de densité rho et épaisseur thick (Constant ou float).
    Retourne un fem.Constant vectoriel.
    """
    # thick peut être fem.Constant ou float
    if isinstance(thick, fem.Constant):
        t_val = thick.value
    else:
        t_val = float(thick)
    gx, gy, gz = g_vec
    return fem.Constant(domain, (rho * t_val * gx,
                                 rho * t_val * gy,
                                 rho * t_val * gz))


def gaussian_iceberg_pressure(domain, center, sigma, p0):
    """
    Pression 'iceberg' comme gaussienne localisée sur la coque.

    center : tuple (xc, yc, zc) position du centre de l'iceberg
    sigma  : largeur de la gaussienne
    p0     : amplitude (pression max, en Pa)
    Retourne une expression UFL scalaire p(x).
    """
    x = ufl.SpatialCoordinate(domain)
    xc, yc, zc = center
    r2 = (x[0] - xc)**2 + (x[1] - yc)**2 + (x[2] - zc)**2
    return p0 * ufl.exp(-r2 / (2 * sigma**2))

def moving_gaussian_pressure(domain, c0, v_ice, t, sigma, p0):
    x = ufl.SpatialCoordinate(domain)
    c = c0 + t * v_ice
    r2 = ufl.dot(x - c, x - c)
    return p0 * ufl.exp(-r2 / (2 * sigma**2))
