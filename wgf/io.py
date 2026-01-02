"""IO utilities for results and GIF outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def canonical_json(data: Any) -> str:
    normalized = _normalize_params(data)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_params(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    for key in ("actual_num_steps", "actual_total_time", "actual_mlp_scale"):
        cleaned.pop(key, None)
    return cleaned


def find_matching_run(results_dir: Path, params_json: str) -> Optional[Path]:
    if not results_dir.exists():
        return None
    for params_path in results_dir.rglob("params.json"):
        try:
            existing = json.loads(params_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if canonical_json(_normalize_params(existing)) == params_json:
            return params_path.parent
    return None


def _format_beta(beta: float) -> str:
    text = f"{beta:.6g}"
    return text.replace("-", "m").replace(".", "p")


def make_run_dir(results_dir: Path, beta: float, params_json: str) -> Path:
    beta_label = _format_beta(beta)
    base = results_dir / f"beta_{beta_label}"
    run_dir = base
    counter = 1
    while run_dir.exists():
        run_dir = results_dir / f"{base.name}__{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, data: Any, compact: bool = False) -> None:
    if compact:
        text = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    else:
        text = json.dumps(data, ensure_ascii=True, indent=2)
    path.write_text(text, encoding="utf-8")


def save_gif(frame_paths: Iterable[Path], output_path: Path, frame_duration_s: float) -> bool:
    paths = list(frame_paths)
    if not paths:
        return False

    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is not None:
        images = [Image.open(path) for path in paths]
        duration_ms = int(round(frame_duration_s * 1000))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        for img in images:
            img.close()
        return True

    try:
        import imageio.v2 as imageio
    except Exception:
        return False

    with imageio.get_writer(output_path, mode="I", duration=frame_duration_s) as writer:
        for path in paths:
            writer.append_data(imageio.imread(path))
    return True


def save_gif_from_images(images: Sequence, output_path: Path, frame_duration_s: float) -> bool:
    if not images:
        return False
    try:
        from PIL import Image
    except Exception:
        return False

    duration_ms = int(round(frame_duration_s * 1000))
    first = images[0]
    if isinstance(first, Image.Image):
        frames = images
    else:
        frames = [Image.fromarray(img) for img in images]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return True
