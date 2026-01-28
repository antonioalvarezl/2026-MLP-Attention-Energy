"""Entry point for running the WGF simulations."""
from __future__ import annotations

import json
from pathlib import Path


def _config_dimension(path: Path) -> tuple[int, bool]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 2, False
    if not isinstance(data, dict):
        return 2, False
    dimension = data.get("dimension", 2)
    if isinstance(dimension, list):
        dims = []
        for value in dimension:
            try:
                dims.append(int(value))
            except (TypeError, ValueError):
                continue
        if not dims:
            return 2, True
        if len(dims) == 1:
            return dims[0], False
        return max(dims), True
    try:
        return int(dimension), False
    except (TypeError, ValueError):
        return 2, False


def main() -> None:
    config_path = Path("config.json")
    dimension, is_list = _config_dimension(config_path)
    if is_list or dimension >= 3:
        from wgf_sphere.runner_S2 import main as run_main
    else:
        from wgf_circle.runner import main as run_main
    run_main(config_path)


if __name__ == "__main__":
    main()
