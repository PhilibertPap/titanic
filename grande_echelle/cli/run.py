try:
    from ..main import run
except ImportError:  # pragma: no cover - script execution fallback
    from main import run


if __name__ == "__main__":
    run()

