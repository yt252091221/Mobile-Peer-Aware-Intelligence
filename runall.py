"""Compatibility alias for the one-click k3 reproduction entrypoint."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "run_all.py"), run_name="__main__")
