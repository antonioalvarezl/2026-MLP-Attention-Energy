"""Entry point for running the WGF simulations."""
from __future__ import annotations

import json
from pathlib import Path


def _config_dimension(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 2
    if not isinstance(data, dict):
        return 2
    try:
        return int(data.get("dimension", 2))
    except (TypeError, ValueError):
        return 2


def main() -> None:
    config_path = Path("config.json")
    dimension = _config_dimension(config_path)
    if dimension == 3:
        from wgf_sphere.runner_S2 import main as run_main
    else:
        from wgf_circle.runner import main as run_main
    run_main(config_path)


if __name__ == "__main__":
    main()
