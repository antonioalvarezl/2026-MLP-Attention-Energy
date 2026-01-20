"""Simulation runner for S2 self-attention + MLP drift simulations."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime
import io
import json
from pathlib import Path
import time
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
from tqdm import tqdm

from wgf_circle.analysis import cluster_threshold
from wgf_circle.io import canonical_json, find_matching_run, make_run_dir, save_gif_from_images, write_json
from wgf_circle.plotting import (
    MLP_COLOR,
    NULL_COLOR,
    SA_COLOR,
    make_cluster_bar_plot,
    make_cluster_bar_plot_with_null,
    save_figure,
)

from .analysis_S2 import cluster_count_s2, cluster_max_spread_s2, convergence_index_s2
from .config_S2 import RunConfig, SeedPlan, build_seed_plan, load_config
from .dynamics_S2 import (
    MLPConfig,
    MLPParams,
    SimulationConfig,
    attention_drift_particles_vectors,
    mlp_drift_vectors,
    sample_mlp_params,
    sample_points_on_sphere,
    simulate_positions,
    step_positions,
)
from .plotting_S2 import (
    SphereMeshCache,
    create_sphere_mesh_cache,
    make_s2_comparison_figure,
    make_s2_histogram_bar_figure,
    make_s2_histogram_comparison_figure,
    make_s2_single_figure,
    make_s2_trajectory_figure,
    s2_histogram_max_count,
    write_s2_interactive_html,
)


def progress_interval(total_steps: int, target_updates: int = 100) -> int:
    if total_steps <= 0:
        return 1
    return max(1, total_steps // max(1, target_updates))


class ProgressHandle:
    def __init__(self, total: int, label: str = "", unit: str = "step") -> None:
        self.total = max(1, int(total))
        self.label = label
        self.unit = unit
        self._last = 0
        self._bar = tqdm(
            total=self.total,
            desc=self.label,
            unit=self.unit,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def update_to(self, current: int, total: Optional[int] = None) -> None:
        if total is not None:
            self.total = max(1, int(total))
            self._bar.total = self.total
        current = min(int(current), self.total)
        delta = current - self._last
        if delta <= 0:
            return
        self._last = current
        self._bar.update(delta)

    def close(self) -> None:
        if self._last < self.total:
            self._bar.update(self.total - self._last)
        self._bar.close()


def iter_progress(iterable: Iterable[int], label: str = "", unit: str = "it") -> Iterator[int]:
    return tqdm(
        iterable,
        desc=label,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


def _frame_indices(
    times_null: np.ndarray,
    times_mlp: np.ndarray,
    interval: float,
    frame_limit: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if times_null.size == 0 or times_mlp.size == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)
    if interval <= 0.0:
        interval = float(times_mlp[-1]) if times_mlp.size > 1 else 1.0
    max_time = float(min(times_null[-1], times_mlp[-1]))
    target_times = np.arange(0.0, max_time + 1e-9, interval)
    if target_times.size == 0:
        target_times = np.array([0.0])
    if abs(target_times[-1] - max_time) > 1e-9:
        target_times = np.append(target_times, max_time)
    null_idx = np.searchsorted(times_null, target_times, side="left")
    mlp_idx = np.searchsorted(times_mlp, target_times, side="left")
    null_idx = np.clip(null_idx, 0, times_null.size - 1)
    mlp_idx = np.clip(mlp_idx, 0, times_mlp.size - 1)
    if frame_limit is not None and frame_limit > 0 and target_times.size > frame_limit:
        sel = np.linspace(0, target_times.size - 1, frame_limit)
        sel = np.round(sel).astype(int)
        target_times = target_times[sel]
        null_idx = null_idx[sel]
        mlp_idx = mlp_idx[sel]
    return target_times, null_idx, mlp_idx


def _frame_times_with_freeze(
    times_null: np.ndarray,
    times_mlp: np.ndarray,
    interval: float,
    frame_limit: Optional[int],
) -> np.ndarray:
    if times_null.size == 0 or times_mlp.size == 0:
        return np.array([])
    if interval <= 0.0:
        max_time = max(float(times_null[-1]), float(times_mlp[-1]))
        interval = max_time if max_time > 1.0 else 1.0
    max_time = float(max(times_null[-1], times_mlp[-1]))
    target_times = np.arange(0.0, max_time + 1e-9, interval)
    if target_times.size == 0:
        target_times = np.array([0.0])
    if abs(target_times[-1] - max_time) > 1e-9:
        target_times = np.append(target_times, max_time)
    if frame_limit is not None and frame_limit > 0 and target_times.size > frame_limit:
        sel = np.linspace(0, target_times.size - 1, frame_limit)
        sel = np.round(sel).astype(int)
        target_times = target_times[sel]
    return target_times


def _interpolate_positions(
    times: np.ndarray,
    history: np.ndarray,
    target_time: float,
) -> np.ndarray:
    if times.size == 0:
        return history[0]
    if target_time <= times[0]:
        return history[0]
    if target_time >= times[-1]:
        return history[-1]
    idx = int(np.searchsorted(times, target_time, side="right") - 1)
    t0 = float(times[idx])
    t1 = float(times[idx + 1])
    if t1 <= t0:
        return history[idx]
    alpha = (target_time - t0) / (t1 - t0)
    p0 = history[idx]
    p1 = history[idx + 1]
    n0 = np.linalg.norm(p0, axis=1, keepdims=True)
    n1 = np.linalg.norm(p1, axis=1, keepdims=True)
    n0_safe = np.where(n0 > 0.0, n0, 1.0)
    n1_safe = np.where(n1 > 0.0, n1, 1.0)
    p0_unit = p0 / n0_safe
    p1_unit = p1 / n1_safe
    dot = np.sum(p0_unit * p1_unit, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    mask = sin_omega > 1e-6
    out = np.empty_like(p0_unit)
    if np.any(mask):
        coeff0 = np.sin((1.0 - alpha) * omega[mask]) / sin_omega[mask]
        coeff1 = np.sin(alpha * omega[mask]) / sin_omega[mask]
        out[mask] = coeff0[:, None] * p0_unit[mask] + coeff1[:, None] * p1_unit[mask]
    if np.any(~mask):
        out[~mask] = (1.0 - alpha) * p0_unit[~mask] + alpha * p1_unit[~mask]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    safe = np.isfinite(norms) & (norms > 1e-8)
    norms_safe = np.where(safe, norms, 1.0)
    normalized = out / norms_safe
    normalized = np.where(safe, normalized, p0_unit)
    return normalized


def _save_s2_evolution_gif(
    output_path: Path,
    times_null: np.ndarray,
    history_null: np.ndarray,
    times_mlp: np.ndarray,
    history_mlp: np.ndarray,
    beta: float,
    n_particles: int,
    interval: float,
    frame_limit: Optional[int],
    null_color: str,
    mlp_color: str,
    potential_params: Optional[tuple[np.ndarray, np.ndarray, str]],
    show_potential: bool,
    point_size: float,
    attention_label: str,
    mlp_title: str,
    ascending: bool = False,
    rotate: bool = True,
    rotation_cycles: float = 1.0,
) -> None:
    frame_times, _, _ = _frame_indices(times_null, times_mlp, interval, frame_limit)
    if frame_times.size == 0:
        return

    try:
        from PIL import Image
    except Exception:
        print("Warning: PIL not available; skipping S2 GIF.")
        return

    # Pre-compute and cache the sphere mesh and potential colors once
    # Since the MLP potential is static, we avoid recomputing it every frame
    mesh_cache = create_sphere_mesh_cache(
        null_params=None,
        mlp_params=potential_params,
        show_potential=show_potential,
    )

    gif_images = []
    writer = None
    try:
        import imageio.v2 as imageio
    except Exception:
        imageio = None
    if imageio is not None:
        try:
            writer = imageio.get_writer(output_path, mode="I", duration=interval)
        except Exception:
            writer = None
    frame_iter = iter_progress(range(len(frame_times)), label=f"GIF {output_path.stem}", unit="frame")
    total_frames = max(1, len(frame_times))
    try:
        for frame_idx in frame_iter:
            t = float(frame_times[frame_idx])
            n_points = _interpolate_positions(times_null, history_null, t)
            m_points = _interpolate_positions(times_mlp, history_mlp, t)
            azim = 35.0
            if rotate:
                azim = 35.0 + 360.0 * rotation_cycles * (frame_idx / total_frames)
            asc_desc = "Ascending" if ascending else "Descending"
            title = (
                rf"$\mathrm{{{attention_label}\ {asc_desc}}},\ t={t:.3f},\ \beta={beta:g},\ "
                rf"N={n_particles},\ {mlp_title}$"
            )
            fig = make_s2_comparison_figure(
                n_points,
                m_points,
                null_color,
                mlp_color,
                title,
                null_params=None,
                mlp_params=potential_params,
                show_potential=show_potential,
                point_size=point_size,
                view_elev=20.0,
                view_azim=azim,
                show_title=True,
                mesh_cache=mesh_cache,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120)
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            if writer is not None:
                writer.append_data(np.asarray(img))
            else:
                gif_images.append(img.copy())
            img.close()
            buf.close()
            import matplotlib.pyplot as plt

            plt.close(fig)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        if not save_gif_from_images(gif_images, output_path, interval):
            print(f"Warning: GIF not generated for {output_path.name}.")
        for img in gif_images:
            try:
                img.close()
            except Exception:
                pass


def _save_s2_histogram_gif(
    output_path: Path,
    times: np.ndarray,
    history: np.ndarray,
    interval: float,
    frame_limit: Optional[int],
    color: str,
    potential_params: Optional[tuple[np.ndarray, np.ndarray, str]],
    show_potential: bool,
    title: str = "",
) -> None:
    frame_times, _, _ = _frame_indices(times, times, interval, frame_limit)
    if frame_times.size == 0:
        return

    try:
        from PIL import Image
    except Exception:
        print("Warning: PIL not available; skipping S2 histogram GIF.")
        return

    bins = 36
    z_max = 0.0
    scale_iter = range(len(frame_times))
    if len(frame_times) > 1:
        scale_iter = iter_progress(
            range(len(frame_times)),
            label=f"Scale {output_path.stem}",
            unit="frame",
        )
    for frame_idx in scale_iter:
        t = float(frame_times[frame_idx])
        points = _interpolate_positions(times, history, t)
        z_max = max(z_max, s2_histogram_max_count(points, bins=bins))
    if z_max <= 0.0:
        z_max = 1.0

    gif_images = []
    writer = None
    try:
        import imageio.v2 as imageio
    except Exception:
        imageio = None
    if imageio is not None:
        try:
            writer = imageio.get_writer(output_path, mode="I", duration=interval)
        except Exception:
            writer = None
    frame_iter = iter_progress(range(len(frame_times)), label=f"GIF {output_path.stem}", unit="frame")
    try:
        for frame_idx in frame_iter:
            t = float(frame_times[frame_idx])
            points = _interpolate_positions(times, history, t)
            fig = make_s2_histogram_bar_figure(
                points,
                color,
                bins=bins,
                potential_params=potential_params,
                show_potential=show_potential,
                z_max=z_max,
                title=title,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120)
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            if writer is not None:
                writer.append_data(np.asarray(img))
            else:
                gif_images.append(img.copy())
            img.close()
            buf.close()
            import matplotlib.pyplot as plt

            plt.close(fig)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        if not save_gif_from_images(gif_images, output_path, interval):
            print(f"Warning: GIF not generated for {output_path.name}.")
        for img in gif_images:
            try:
                img.close()
            except Exception:
                pass


def _save_s2_histogram_comparison_gif(
    output_path: Path,
    times_null: np.ndarray,
    history_null: np.ndarray,
    times_mlp: np.ndarray,
    history_mlp: np.ndarray,
    interval: float,
    frame_limit: Optional[int],
    null_color: str,
    mlp_color: str,
    potential_params: Optional[tuple[np.ndarray, np.ndarray, str]],
    show_potential: bool,
    attention_label: str,
    mlp_title: str,
    beta: float,
    n_particles: int,
    ascending: bool = False,
) -> None:
    frame_times = _frame_times_with_freeze(times_null, times_mlp, interval, frame_limit)
    if frame_times.size == 0:
        return

    try:
        from PIL import Image
    except Exception:
        print("Warning: PIL not available; skipping S2 histogram GIF.")
        return

    bins = 36
    z_max = 0.0
    scale_iter = range(len(frame_times))
    if len(frame_times) > 1:
        scale_iter = iter_progress(
            range(len(frame_times)),
            label=f"Scale {output_path.stem}",
            unit="frame",
        )
    for frame_idx in scale_iter:
        t = float(frame_times[frame_idx])
        null_points = _interpolate_positions(times_null, history_null, t)
        mlp_points = _interpolate_positions(times_mlp, history_mlp, t)
        z_max = max(
            z_max,
            s2_histogram_max_count(null_points, bins=bins),
            s2_histogram_max_count(mlp_points, bins=bins),
        )
    if z_max <= 0.0:
        z_max = 1.0

    gif_images = []
    writer = None
    try:
        import imageio.v2 as imageio
    except Exception:
        imageio = None
    if imageio is not None:
        try:
            writer = imageio.get_writer(output_path, mode="I", duration=interval)
        except Exception:
            writer = None
    frame_iter = iter_progress(range(len(frame_times)), label=f"GIF {output_path.stem}", unit="frame")
    try:
        for frame_idx in frame_iter:
            t = float(frame_times[frame_idx])
            null_points = _interpolate_positions(times_null, history_null, t)
            mlp_points = _interpolate_positions(times_mlp, history_mlp, t)
            asc_desc = "Ascending" if ascending else "Descending"
            title = (
                rf"$\mathrm{{{attention_label}\ {asc_desc}}},\ t={t:.3f},\ \beta={beta:g},\ "
                rf"N={n_particles},\ {mlp_title}$"
            )
            fig = make_s2_histogram_comparison_figure(
                null_points,
                mlp_points,
                null_color,
                mlp_color,
                title=title,
                bins=bins,
                null_params=None,
                mlp_params=potential_params,
                show_null_potential=False,
                show_mlp_potential=show_potential,
                z_max=z_max,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120)
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            if writer is not None:
                writer.append_data(np.asarray(img))
            else:
                gif_images.append(img.copy())
            img.close()
            buf.close()
            import matplotlib.pyplot as plt

            plt.close(fig)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        if not save_gif_from_images(gif_images, output_path, interval):
            print(f"Warning: GIF not generated for {output_path.name}.")
        for img in gif_images:
            try:
                img.close()
            except Exception:
                pass


def _build_params_dict(
    config: RunConfig,
    beta: float,
    seeds: SeedPlan,
    num_steps: int,
    effective_total_time: Optional[float],
    mlp_scale: Optional[float] = None,
    mlp_scale_mode: Optional[str] = None,
    actual_num_steps: Optional[int] = None,
    actual_total_time: Optional[float] = None,
    actual_mlp_scale: Optional[float] = None,
) -> dict:
    scale_value = config.mlp_scale if mlp_scale is None else mlp_scale
    scale_mode_value = config.mlp_scale_mode if mlp_scale_mode is None else mlp_scale_mode
    if np.isinf(config.total_time):
        total_time_value = "inf"
        num_steps_value = None
        effective_value = None
        max_steps_value = config.max_steps
    else:
        total_time_value = float(config.total_time)
        num_steps_value = num_steps
        effective_value = effective_total_time
        max_steps_value = None

    params = {
        "beta": beta,
        "dimension": config.dimension,
        "n_particles": config.n_particles,
        "dt": config.dt,
        "total_time": total_time_value,
        "effective_total_time": effective_value,
        "num_steps": num_steps_value,
        "max_steps": max_steps_value,
        "actual_num_steps": actual_num_steps,
        "actual_total_time": actual_total_time,
        "save_every": config.save_every,
        "k_max": config.k_max,
        "num_mlp_inits": config.num_mlp_inits,
        "mlp_units": config.mlp_units,
        "mlp_scale": scale_value,
        "mlp_scale_mode": scale_mode_value,
        "actual_mlp_scale": actual_mlp_scale,
        "activation": config.activation,
        "gradient_MLP": config.gradient_mlp,
        "particle_seed": config.particle_seed,
        "mlp_seed": config.mlp_seed,
        "particle_seeds": seeds.particle_seeds,
        "mlp_seeds": seeds.mlp_seeds,
        "plot_interval": config.plot_interval,
        "num_point_inits": config.num_point_inits,
        "cluster_scale": config.cluster_scale,
        "mass_threshold": config.mass_threshold,
        "convergence_window": config.convergence_window,
        "convergence_drift_tol": config.convergence_drift_tol,
        "convergence_spread_factor": config.convergence_spread_factor,
        "attention_mode": config.attention_mode,
        "integrator": config.integrator,
        "self_attention": config.self_attention,
        "ascending": config.ascending,
        "output_frame_limit": config.output_frame_limit,
        "mlp0_output_frame_limit": config.mlp0_output_frame_limit,
        "gif_sphere": config.gif_sphere,
        "gif_histogram": config.gif_histogram,
        "sphere_html_view": config.sphere_html_view,
        "sphere_gif_rotations": config.sphere_gif_rotations,
    }
    if config.mlp_params_path is not None:
        params["mlp_params_path"] = str(config.mlp_params_path)
        params["mlp_params_hash"] = config.mlp_params_hash
    return params


def _write_run_summary(
    run_dir: Path,
    beta: float,
    params_json: str,
    null_counts: list[int],
    mlp_counts: list[int],
    null_mode_counts: list[int],
    mlp_mode_counts: list[int],
    null_mass_counts: list[int],
    mlp_mass_counts: list[int],
    null_stop_reasons: list[str],
    mlp_stop_reasons: list[str],
    num_mlp_inits: int,
    num_point_inits: int,
    runtime_seconds: float,
    mlp_scale: float,
    mlp_scale_mode: str,
    null_cluster_times: list[Optional[float]],
    mlp_cluster_times: list[Optional[float]],
    histogram_edges: list[float],
    null_histogram_densities: list[list[float]],
    mlp_histogram_densities: list[list[float]],
) -> None:
    summary = {
        "beta": beta,
        "sqrt_beta": float(np.sqrt(beta)),
        "params_json": params_json,
        "null_counts": null_counts,
        "mlp_counts": mlp_counts,
        "null_mode_counts": null_mode_counts,
        "mlp_mode_counts": mlp_mode_counts,
        "null_mass_counts": null_mass_counts,
        "mlp_mass_counts": mlp_mass_counts,
        "null_stop_reasons": null_stop_reasons,
        "mlp_stop_reasons": mlp_stop_reasons,
        "num_mlp_inits": num_mlp_inits,
        "num_point_inits": num_point_inits,
        "runtime_seconds": runtime_seconds,
        "mlp_scale": mlp_scale,
        "mlp_scale_mode": mlp_scale_mode,
        "null_cluster_times": null_cluster_times,
        "mlp_cluster_times": mlp_cluster_times,
        "histogram_edges": histogram_edges,
        "null_histogram_densities": null_histogram_densities,
        "mlp_histogram_densities": mlp_histogram_densities,
    }
    write_json(run_dir / "summary.json", summary, compact=False)


def _load_run_summaries(
    experiment_dir: Path,
    expected_params: dict[float, str],
) -> list[dict]:
    summaries: dict[float, tuple[dict, float]] = {}
    if not experiment_dir.exists():
        return []
    for summary_path in experiment_dir.rglob("summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        beta = data.get("beta")
        params_json = data.get("params_json")
        if beta is None or params_json is None:
            continue
        try:
            beta_value = float(beta)
        except (TypeError, ValueError):
            continue
        expected = expected_params.get(beta_value)
        if expected is None:
            continue
        try:
            normalized = canonical_json(json.loads(params_json))
        except Exception:
            continue
        if normalized != expected:
            continue
        mtime = summary_path.stat().st_mtime
        existing = summaries.get(beta_value)
        if existing is None or mtime > existing[1]:
            summaries[beta_value] = (data, mtime)

    ordered = [item[0] for item in summaries.values()]
    ordered.sort(key=lambda entry: float(entry.get("sqrt_beta", np.sqrt(float(entry["beta"])))))
    return ordered


def _load_scale_summaries(
    experiment_dir: Path,
    expected_params: dict[float, str],
) -> list[tuple[float, dict, Path]]:
    if not experiment_dir.exists():
        return []
    expected_by_params = {params_json: scale for scale, params_json in expected_params.items()}
    entries: list[tuple[float, dict, Path]] = []
    for summary_path in experiment_dir.rglob("summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        params_json = data.get("params_json")
        if params_json is None:
            continue
        scale = expected_by_params.get(params_json)
        if scale is None:
            continue
        entries.append((scale, data, summary_path.parent))
    entries.sort(key=lambda item: item[0])
    return entries


def _simulate_until_convergence_s2(
    x0: np.ndarray,
    sim_config: SimulationConfig,
    mlp_params: Optional[MLPParams],
    threshold: float,
    convergence_window: int,
    max_steps: int,
    drift_tol: float,
    spread_factor: float,
    progress: Optional[Callable[[int, int], None]] = None,
    progress_every: Optional[int] = None,
    save_history: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Simulate until convergence. If save_history=False, only keeps initial and final states."""
    x = x0.astype(np.float64, copy=True)
    times = [0.0]
    history = [x.copy()]
    counts = deque(maxlen=convergence_window)
    if progress is not None:
        if progress_every is None:
            progress_every = max(1, max_steps // 100)
        progress(0, max_steps)

    for step in range(1, max_steps + 1):
        x = step_positions(x, sim_config, mlp_params)

        if step % sim_config.save_every == 0 or step == max_steps:
            if save_history:
                times.append(step * sim_config.dt)
                history.append(x.copy())
            count = cluster_count_s2(x, threshold)
            counts.append(count)
            max_spread = cluster_max_spread_s2(x, threshold)
            spread_ok = spread_factor <= 0.0 or max_spread <= spread_factor * threshold
            if drift_tol <= 0.0:
                drift_ok = True
            else:
                drift_check = attention_drift_particles_vectors(
                    x,
                    sim_config.beta,
                    sim_config.attention_mode,
                    self_attention=sim_config.self_attention,
                    ascending=sim_config.ascending,
                )
                if mlp_params is not None:
                    drift_check += mlp_drift_vectors(x, mlp_params)
                drift_norm = np.linalg.norm(drift_check, axis=1)
                drift_ok = float(np.max(drift_norm)) <= drift_tol
            if len(counts) == convergence_window and len(set(counts)) == 1 and spread_ok and drift_ok:
                if progress is not None:
                    progress(step, max_steps)
                if not save_history:
                    times.append(step * sim_config.dt)
                    history.append(x.copy())
                return np.asarray(times), np.asarray(history), step, "convergence"

        if progress is not None and (step % progress_every == 0 or step == max_steps):
            progress(step, max_steps)

    if not save_history:
        times.append(max_steps * sim_config.dt)
        history.append(x.copy())
    return np.asarray(times), np.asarray(history), max_steps, "max_steps"


def run_experiment_s2(config: RunConfig) -> None:
    is_infinite = np.isinf(config.total_time)
    if is_infinite:
        num_steps = int(config.max_steps)
    else:
        num_steps = int(round(config.total_time / config.dt))
    print("Run configuration (S2 stats-only):")
    print(f"  betas={config.betas}")
    print(f"  n_particles={config.n_particles}")
    print(f"  dt={config.dt}")
    if is_infinite:
        print("  total_time=inf (auto-stop on convergence)")
        print(f"  max_steps={config.max_steps}")
    else:
        print(f"  total_time={config.total_time}")
    print(f"  num_steps={num_steps}")
    print(f"  save_every={config.save_every}")
    print(f"  attention_mode={config.attention_mode}")
    print(f"  integrator={config.integrator}")
    print(f"  activation={config.activation}")
    print(f"  mlp_units={config.mlp_units}")
    print(f"  mlp_scale={config.mlp_scales}")
    print(f"  mlp_scale_mode={config.mlp_scale_mode}")
    print(f"  num_mlp_inits={config.num_mlp_inits}")
    print(f"  num_point_inits={config.num_point_inits}")
    print(f"  particle_seed={config.particle_seed}")
    print(f"  mlp_seed={config.mlp_seed}")
    if config.mlp_params_path is not None:
        print(f"  mlp_params_path={config.mlp_params_path}")
    print(f"  sphere_html_view={config.sphere_html_view}")
    print(f"  sphere_gif_rotations={config.sphere_gif_rotations}")
    print(f"  cluster_scale={config.cluster_scale}")
    print(f"  mass_threshold={config.mass_threshold}")
    print(f"  convergence_window={config.convergence_window}")
    print(f"  convergence_drift_tol={config.convergence_drift_tol}")
    print(f"  convergence_spread_factor={config.convergence_spread_factor}")
    print(f"  self_attention={config.self_attention}")
    print(f"  ascending={config.ascending}")
    print(f"  output_frame_limit={config.output_frame_limit}")
    print(f"  mlp0_output_frame_limit={config.mlp0_output_frame_limit}")
    print(f"  gif_sphere={config.gif_sphere}")
    print(f"  gif_histogram={config.gif_histogram}")

    seed_plan = build_seed_plan(config)
    if num_steps <= 0:
        raise ValueError("total-time must be positive.")

    effective_total_time = None if is_infinite else num_steps * config.dt
    progress_every = progress_interval(num_steps)
    experiment_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.experiment_dir is None:
        experiment_dir = config.results_dir / f"experiment_{experiment_stamp}"
        experiment_dir.mkdir(parents=True, exist_ok=False)
    else:
        experiment_dir = config.experiment_dir
        if not experiment_dir.is_absolute():
            if experiment_dir.parent == Path("."):
                experiment_dir = config.results_dir / experiment_dir
        experiment_dir.mkdir(parents=True, exist_ok=True)

    expected_params: dict[float, str] = {}
    mlp_scales = config.mlp_scales
    multi_scale = len(mlp_scales) > 1
    sweep_expected_params: dict[float, str] = {}
    mlp_color = MLP_COLOR if config.attention_mode == "unnormalized" else SA_COLOR
    attention_label = "USA" if config.attention_mode == "unnormalized" else "SA"

    for beta in config.betas:
        print(f"Starting beta={beta}")
        seeds = seed_plan[beta]

        scale_runs: list[tuple[float, str]] = []
        expected_scale_params: dict[float, str] = {}
        for mlp_scale in mlp_scales:
            params = _build_params_dict(
                config,
                beta,
                seeds,
                num_steps,
                effective_total_time,
                mlp_scale=mlp_scale,
                mlp_scale_mode=config.mlp_scale_mode,
            )
            params_json = canonical_json(params)
            expected_scale_params[mlp_scale] = params_json
            if not multi_scale:
                expected_params[beta] = params_json

            existing = find_matching_run(experiment_dir, params_json)
            if existing is not None:
                print(
                    f"Skipping beta={beta}, mlp_scale={mlp_scale}: params already exist at {existing}"
                )
                continue
            scale_runs.append((mlp_scale, params_json))

        if multi_scale:
            sweep_expected_params = expected_scale_params

        if not scale_runs:
            continue

        sim_config = SimulationConfig(
            beta=beta,
            dt=config.dt,
            num_steps=num_steps,
            save_every=config.save_every,
            attention_mode=config.attention_mode,
            self_attention=config.self_attention,
            ascending=config.ascending,
            integrator=config.integrator,
        )

        points0_list = []
        for seed in seeds.particle_seeds:
            rng_particles = np.random.default_rng(seed)
            points0_list.append(
                sample_points_on_sphere(rng_particles, config.n_particles, config.dimension)
            )

        threshold = cluster_threshold(beta, config.cluster_scale)
        match_null_to_mlp = (
            is_infinite
            and config.gradient_mlp
            and not config.ascending
            and config.dimension == 3
        )

        # Only keep first history for plots; count clusters immediately for others
        null_steps = []
        null_stop_reasons = []
        null_cluster_times: list[Optional[float]] = []
        null_counts = []
        null_did_not_converge = False
        example_initial = points0_list[0]
        example_null_final = None
        example_null_times = None
        example_null_hist = None
        need_full_history = config.gif_sphere or config.gif_histogram or config.pdf_trajectory

        if not match_null_to_mlp:
            for idx, points0 in enumerate(points0_list):
                # Only save full history for first run if needed for GIFs
                save_this_history = (idx == 0) and need_full_history
                label = f"beta={beta} MLP_null init {idx + 1}/{config.num_point_inits}"
                bar = ProgressHandle(num_steps, label=label)
                if is_infinite:
                    times, points_hist, step_count, stop_reason = _simulate_until_convergence_s2(
                        points0,
                        sim_config,
                        None,
                        threshold,
                        config.convergence_window,
                        config.max_steps,
                        config.convergence_drift_tol,
                        config.convergence_spread_factor,
                        progress=bar.update_to,
                        progress_every=progress_every,
                        save_history=save_this_history,
                    )
                    if stop_reason != "convergence":
                        null_did_not_converge = True
                else:
                    times, points_hist = simulate_positions(
                        points0,
                        sim_config,
                        None,
                        progress=bar.update_to,
                        progress_every=progress_every,
                        save_history=save_this_history,
                    )
                    step_count = num_steps
                    stop_reason = "fixed_time"
                bar.close()
                null_steps.append(step_count)
                null_stop_reasons.append(stop_reason)
                null_cluster_times.append(
                    step_count * config.dt if stop_reason == "convergence" else None
                )
                print(f"  [null init {idx + 1}] stop_reason={stop_reason}")

                # Count clusters immediately
                if is_infinite:
                    conv_idx = len(points_hist) - 1
                else:
                    conv_idx = convergence_index_s2(points_hist, threshold, config.convergence_window)
                null_counts.append(cluster_count_s2(points_hist[conv_idx], threshold))

                # Only keep first history for example plots
                if idx == 0:
                    example_null_final = points_hist[-1]
                    example_null_times = times
                    example_null_hist = points_hist if need_full_history else None
                # Discard history for non-first runs to save memory
                del points_hist, times

        for mlp_scale_eff, params_json in scale_runs:
            run_dir = make_run_dir(experiment_dir, beta, params_json)
            print(f"Run directory: {run_dir}")
            run_start = time.perf_counter()
            did_not_converge = False if match_null_to_mlp else null_did_not_converge
            all_steps: list[int] = []
            if not match_null_to_mlp:
                all_steps.extend(null_steps)
            mlp_std_label = f"{mlp_scale_eff:.3g}"
            if config.mlp_scale_mode == "norm":
                mlp_title = rf"\|\omega\|\,=\,{mlp_std_label}"
            else:
                mlp_title = rf"\mathrm{{std(MLP)}}\,=\,{mlp_std_label}"

            mlp_config = MLPConfig(
                n_units=config.mlp_units,
                activation=config.activation,
                weight_scale=mlp_scale_eff,
                weight_scale_mode=config.mlp_scale_mode,
                gradient_mlp=config.gradient_mlp,
                dimension=config.dimension,
            )

            mlp_params_list = []
            if config.mlp_params is not None:
                for a, omega, activation in config.mlp_params:
                    mlp_params_list.append(
                        MLPParams(
                            a=a.astype(np.float64),
                            omega=omega.astype(np.float64),
                            activation=activation,
                        )
                    )
            else:
                for mlp_seed in seeds.mlp_seeds:
                    rng_mlp = np.random.default_rng(mlp_seed)
                    mlp_params_list.append(sample_mlp_params(rng_mlp, mlp_config))

            mlp_counts = []
            mlp_stop_reasons = []
            mlp_cluster_times: list[Optional[float]] = []
            example_mlp_final = None
            example_mlp_params = None
            example_mlp_times = None
            example_mlp_hist = None
            mlp_steps_by_point = [None] * len(points0_list) if match_null_to_mlp else None
            for i, mlp_params in enumerate(mlp_params_list):
                if i == 0 and example_mlp_params is None:
                    example_mlp_params = mlp_params
                for j, points0 in enumerate(points0_list):
                    # Only save full history for first MLP run if needed for GIFs
                    save_this_history = (i == 0 and j == 0) and need_full_history
                    label = f"beta={beta} MLP{i + 1} init {j + 1}/{config.num_point_inits}"
                    bar = ProgressHandle(num_steps, label=label)
                    if is_infinite:
                        times, points_hist, step_count, stop_reason = _simulate_until_convergence_s2(
                            points0,
                            sim_config,
                            mlp_params,
                            threshold,
                            config.convergence_window,
                            config.max_steps,
                            config.convergence_drift_tol,
                            config.convergence_spread_factor,
                            progress=bar.update_to,
                            progress_every=progress_every,
                            save_history=save_this_history,
                        )
                        if stop_reason != "convergence":
                            did_not_converge = True
                        if match_null_to_mlp and mlp_steps_by_point is not None and i == 0:
                            mlp_steps_by_point[j] = step_count
                    else:
                        times, points_hist = simulate_positions(
                            points0,
                            sim_config,
                            mlp_params,
                            progress=bar.update_to,
                            progress_every=progress_every,
                            save_history=save_this_history,
                        )
                        step_count = num_steps
                        stop_reason = "fixed_time"
                    bar.close()
                    all_steps.append(step_count)
                    mlp_stop_reasons.append(stop_reason)
                    mlp_cluster_times.append(
                        step_count * config.dt if stop_reason == "convergence" else None
                    )
                    
                    # Count clusters immediately
                    if is_infinite:
                        conv_idx = len(points_hist) - 1
                    else:
                        conv_idx = convergence_index_s2(points_hist, threshold, config.convergence_window)
                    mlp_counts.append(cluster_count_s2(points_hist[conv_idx], threshold))
                    
                    # Only keep first history for example plots
                    if i == 0 and j == 0:
                        example_mlp_final = points_hist[-1]
                        example_mlp_times = times
                        example_mlp_hist = points_hist if need_full_history else None
                    # Discard history for non-first runs to save memory
                    del points_hist, times

            if match_null_to_mlp:
                run_null_steps = []
                run_null_stop_reasons = []
                run_null_cluster_times: list[Optional[float]] = []
                run_null_counts = []
                for idx, points0 in enumerate(points0_list):
                    # Only save full history for first run if needed for GIFs
                    save_this_history = (idx == 0) and need_full_history
                    target_steps = (
                        mlp_steps_by_point[idx]
                        if mlp_steps_by_point is not None and mlp_steps_by_point[idx] is not None
                        else num_steps
                    )
                    label = f"beta={beta} MLP_null init {idx + 1}/{config.num_point_inits}"
                    bar = ProgressHandle(target_steps, label=label)
                    fixed_config = replace(sim_config, num_steps=target_steps)
                    fixed_progress_every = progress_interval(target_steps)
                    times, points_hist = simulate_positions(
                        points0,
                        fixed_config,
                        None,
                        progress=bar.update_to,
                        progress_every=fixed_progress_every,
                        save_history=save_this_history,
                    )
                    stop_reason = "mlp_time"
                    bar.close()
                    run_null_steps.append(target_steps)
                    run_null_stop_reasons.append(stop_reason)
                    run_null_cluster_times.append(None)
                    print(f"  [null init {idx + 1}] stop_reason={stop_reason}")

                    # Count clusters immediately
                    conv_idx = len(points_hist) - 1
                    run_null_counts.append(cluster_count_s2(points_hist[conv_idx], threshold))

                    # Only keep first history for example plots
                    if idx == 0:
                        example_null_final = points_hist[-1]
                        example_null_times = times
                        example_null_hist = points_hist if need_full_history else None
                    # Discard history for non-first runs to save memory
                    del points_hist, times

                all_steps.extend(run_null_steps)
            else:
                run_null_steps = null_steps
                run_null_stop_reasons = null_stop_reasons
                run_null_cluster_times = null_cluster_times
                run_null_counts = null_counts
                # example_null_* already set above

            actual_num_steps = max(all_steps) if (is_infinite and all_steps) else None
            actual_total_time = actual_num_steps * config.dt if actual_num_steps is not None else None
            params = _build_params_dict(
                config,
                beta,
                seeds,
                num_steps,
                effective_total_time,
                mlp_scale=mlp_scale_eff,
                mlp_scale_mode=config.mlp_scale_mode,
                actual_num_steps=actual_num_steps,
                actual_total_time=actual_total_time,
                actual_mlp_scale=mlp_scale_eff,
            )
            write_json(run_dir / "params.json", params, compact=False)
            run_seconds = time.perf_counter() - run_start
            _write_run_summary(
                run_dir,
                beta,
                params_json,
                run_null_counts,
                mlp_counts,
                [],
                [],
                [],
                [],
                run_null_stop_reasons,
                mlp_stop_reasons,
                config.num_mlp_inits,
                config.num_point_inits,
                run_seconds,
                mlp_scale_eff,
                config.mlp_scale_mode,
                run_null_cluster_times,
                mlp_cluster_times,
                [],
                [],
                [],
            )

            if example_mlp_final is not None:
                potential_params = None
                if config.gradient_mlp and example_mlp_params is not None:
                    potential_params = (
                        example_mlp_params.a,
                        example_mlp_params.omega,
                        example_mlp_params.activation,
                    )
                fig = make_s2_single_figure(
                    example_initial,
                    NULL_COLOR,
                    potential_params=None,
                    show_potential=False,
                    point_size=10.0,
                )
                save_figure(fig, run_dir / "sphere_init_null", formats=("pdf",))
                import matplotlib.pyplot as plt

                plt.close(fig)
                fig = make_s2_single_figure(
                    example_initial,
                    mlp_color,
                    potential_params=potential_params,
                    show_potential=config.gradient_mlp,
                    point_size=10.0,
                )
                save_figure(fig, run_dir / "sphere_init_mlp", formats=("pdf",))
                plt.close(fig)
                mid_null = example_null_hist[len(example_null_hist) // 2]
                mid_mlp = (
                    example_mlp_hist[len(example_mlp_hist) // 2]
                    if example_mlp_hist is not None
                    else example_null_hist[len(example_null_hist) // 2]
                )
                fig = make_s2_single_figure(
                    mid_null,
                    NULL_COLOR,
                    potential_params=None,
                    show_potential=False,
                    point_size=10.0,
                )
                save_figure(fig, run_dir / "sphere_middle_null", formats=("pdf",))
                plt.close(fig)
                fig = make_s2_single_figure(
                    mid_mlp,
                    mlp_color,
                    potential_params=potential_params,
                    show_potential=config.gradient_mlp,
                    point_size=10.0,
                )
                save_figure(fig, run_dir / "sphere_middle_mlp", formats=("pdf",))
                plt.close(fig)
                fig = make_s2_single_figure(
                    example_null_final,
                    NULL_COLOR,
                    potential_params=None,
                    show_potential=False,
                    point_size=10.0,
                )
                save_figure(fig, run_dir / "sphere_final_null", formats=("pdf",))
                plt.close(fig)
                fig = make_s2_single_figure(
                    example_mlp_final,
                    mlp_color,
                    potential_params=potential_params,
                    show_potential=config.gradient_mlp,
                    point_size=10.0,
                )
                save_figure(fig, run_dir / "sphere_final_mlp", formats=("pdf",))
                plt.close(fig)
                fig = make_s2_histogram_bar_figure(
                    example_null_final,
                    NULL_COLOR,
                    potential_params=None,
                    show_potential=False,
                )
                save_figure(fig, run_dir / "sphere_histogram_null", formats=("pdf",))
                plt.close(fig)
                # Without decision boundaries
                fig = make_s2_histogram_bar_figure(
                    example_mlp_final,
                    mlp_color,
                    potential_params=potential_params,
                    show_potential=config.gradient_mlp,
                    show_decision_boundaries=False,
                )
                save_figure(fig, run_dir / "sphere_histogram_mlp", formats=("pdf",))
                plt.close(fig)
                # With decision boundaries
                fig = make_s2_histogram_bar_figure(
                    example_mlp_final,
                    mlp_color,
                    potential_params=potential_params,
                    show_potential=config.gradient_mlp,
                    show_decision_boundaries=True,
                )
                save_figure(fig, run_dir / "sphere_histogram_mlp_boundaries", formats=("pdf",))
                plt.close(fig)
                if config.pdf_trajectory and example_null_hist is not None and example_null_times is not None:
                    # Linear scale
                    fig = make_s2_trajectory_figure(
                        example_null_hist,
                        example_null_times,
                        NULL_COLOR,
                        time_scale="linear",
                    )
                    save_figure(fig, run_dir / "sphere_trajectory_null", formats=("pdf",))
                    plt.close(fig)
                    # Log scale
                    fig = make_s2_trajectory_figure(
                        example_null_hist,
                        example_null_times,
                        NULL_COLOR,
                        time_scale="log",
                    )
                    save_figure(fig, run_dir / "sphere_trajectory_null_log", formats=("pdf",))
                    plt.close(fig)
                if config.pdf_trajectory and example_mlp_hist is not None and example_mlp_times is not None:
                    # Linear scale
                    fig = make_s2_trajectory_figure(
                        example_mlp_hist,
                        example_mlp_times,
                        mlp_color,
                        time_scale="linear",
                    )
                    save_figure(fig, run_dir / "sphere_trajectory_mlp", formats=("pdf",))
                    plt.close(fig)
                    # Log scale
                    fig = make_s2_trajectory_figure(
                        example_mlp_hist,
                        example_mlp_times,
                        mlp_color,
                        time_scale="log",
                    )
                    save_figure(fig, run_dir / "sphere_trajectory_mlp_log", formats=("pdf",))
                    plt.close(fig)
                # Interactive HTML with both MLP and null views
                write_s2_interactive_html(
                    run_dir / "sphere_views.html",
                    initial=example_initial,
                    middle=mid_mlp,
                    final=example_mlp_final,
                    color=mlp_color,
                    mlp_params=potential_params,
                    show_potential=config.gradient_mlp,
                    point_size=5.0,
                    initial_null=example_initial,
                    middle_null=mid_null,
                    final_null=example_null_final,
                    null_color=NULL_COLOR,
                )
                null_frame_limit = config.output_frame_limit
                if config.gradient_mlp and config.mlp0_output_frame_limit is not None:
                    null_frame_limit = min(null_frame_limit, config.mlp0_output_frame_limit)
                evolution_frame_limit = null_frame_limit
                mlp_frame_limit = config.output_frame_limit
                if (
                    config.gif_sphere
                    and example_mlp_times is not None
                    and example_mlp_hist is not None
                ):
                    _save_s2_evolution_gif(
                        run_dir / "sphere_evolution.gif",
                        example_null_times,
                        example_null_hist,
                        example_mlp_times,
                        example_mlp_hist,
                        beta,
                        config.n_particles,
                        config.plot_interval,
                        evolution_frame_limit,
                        NULL_COLOR,
                        mlp_color,
                        potential_params,
                        config.gradient_mlp,
                        10.0,
                        attention_label,
                        mlp_title,
                        ascending=config.ascending,
                        rotate=True,
                        rotation_cycles=config.sphere_gif_rotations,
                    )

                if (
                    config.gif_histogram
                    and example_null_times is not None
                    and example_null_hist is not None
                ):
                    if example_mlp_times is not None and example_mlp_hist is not None:
                        hist_frame_limit = min(null_frame_limit, mlp_frame_limit)
                        _save_s2_histogram_comparison_gif(
                            run_dir / "sphere_histogram.gif",
                            example_null_times,
                            example_null_hist,
                            example_mlp_times,
                            example_mlp_hist,
                            config.plot_interval,
                            hist_frame_limit,
                            NULL_COLOR,
                            mlp_color,
                            potential_params,
                            config.gradient_mlp,
                            attention_label,
                            mlp_title,
                            beta,
                            config.n_particles,
                            ascending=config.ascending,
                        )

            if did_not_converge:
                reasons = sorted(
                    {
                        reason
                        for reason in run_null_stop_reasons + mlp_stop_reasons
                        if reason != "convergence"
                    }
                )
                if reasons:
                    print(f"Warning: convergence not reached (stop_reasons={', '.join(reasons)}).")
                else:
                    print("Warning: convergence not reached before max_steps.")

            print(f"Saved results to {run_dir}")

            # --- Memory cleanup after each run ---
            if example_mlp_hist is not None:
                del example_mlp_hist
            if example_null_hist is not None:
                del example_null_hist
            example_mlp_hist = example_mlp_times = example_mlp_final = None
            example_null_hist = example_null_times = example_null_final = None

        # --- Memory cleanup after all runs for this beta ---
        del points0_list
        points0_list = None
        import gc
        gc.collect()

    if multi_scale and sweep_expected_params:
        entries = _load_scale_summaries(experiment_dir, sweep_expected_params)
        if entries:
            payload_entries = []
            for scale, data, run_dir in entries:
                entry = dict(data)
                entry["run_dir"] = str(run_dir)
                payload_entries.append(entry)
            payload = {
                "beta": config.betas[0] if config.betas else None,
                "mlp_scale_mode": config.mlp_scale_mode,
                "entries": payload_entries,
            }
            write_json(experiment_dir / "mlp_scale_sweep.json", payload, compact=False)
        return

    summaries = _load_run_summaries(experiment_dir, expected_params)
    if summaries:
        sqrt_betas = np.asarray([float(np.sqrt(float(entry["beta"]))) for entry in summaries])
        order = np.argsort(sqrt_betas)
        sqrt_betas = sqrt_betas[order]
        null_counts = [entry.get("null_counts", []) for entry in summaries]
        mlp_counts = [entry.get("mlp_counts", []) for entry in summaries]

        def _stats(values):
            means = []
            stds = []
            for idx in order:
                data = np.asarray(values[idx], dtype=float)
                means.append(float(np.mean(data)) if data.size else 0.0)
                stds.append(float(np.std(data)) if data.size else 0.0)
            return np.asarray(means), np.asarray(stds)

        null_mean, _ = _stats(null_counts)
        mlp_mean, _ = _stats(mlp_counts)
        stats_dir = experiment_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        fig = make_cluster_bar_plot(
            sqrt_betas,
            mlp_mean,
            ylabel="cluster count",
        )
        save_figure(fig, stats_dir / "cluster_count", formats=("pdf",))
        import matplotlib.pyplot as plt

        plt.close(fig)
        fig = make_cluster_bar_plot_with_null(
            sqrt_betas,
            null_mean,
            mlp_mean,
            ylabel="cluster count",
        )
        save_figure(fig, stats_dir / "cluster_count_with_null", formats=("pdf",))
        plt.close(fig)


def main(config_path: Optional[Path] = None) -> None:
    path = config_path or Path("config.json")
    config = load_config(path)
    run_experiment_s2(config)


if __name__ == "__main__":
    main()
