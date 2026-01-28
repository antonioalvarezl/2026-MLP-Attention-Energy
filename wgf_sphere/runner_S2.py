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
    POTENTIAL_NEG_COLOR,
    POTENTIAL_POS_COLOR,
    make_cluster_bar_plot,
    make_cluster_bar_plot_with_null,
    save_figure,
)

from .analysis_S2 import cluster_count_s2, cluster_masses_s2, cluster_max_spread_s2, convergence_index_s2, heaviest_cluster_mass_s2
from .config_S2 import RunConfig, SeedPlan, build_seed_plan, load_config
from .dynamics_S2 import (
    MLPConfig,
    MLPParams,
    SimulationConfig,
    attention_drift_particles_vectors,
    compute_total_energy,
    mlp_drift_vectors,
    sample_mlp_params,
    sample_points_on_sphere,
    simulate_positions,
    step_positions,
)
from .plotting_S2 import (
    SphereMeshCache,
    compute_c_theta,
    create_sphere_mesh_cache,
    make_all_masses_figure,
    make_energy_figure,
    make_heaviest_mass_figure,
    make_mlp_potential_surface_figure,
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


def _resolve_experiment_dir(config: RunConfig, experiment_stamp: str) -> Path:
    if config.experiment_dir is None:
        experiment_dir = config.results_dir / f"experiment_{experiment_stamp}"
        experiment_dir.mkdir(parents=True, exist_ok=False)
        return experiment_dir
    experiment_dir = config.experiment_dir
    if not experiment_dir.is_absolute():
        if experiment_dir.parent == Path("."):
            experiment_dir = config.results_dir / experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _run_multi_dim_experiment(config: RunConfig) -> None:
    experiment_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = _resolve_experiment_dir(config, experiment_stamp)
    dimension_dirs: dict[int, Path] = {}
    for dim in config.dimensions:
        dim_dir = base_dir / f"d{dim}"
        dim_dir.mkdir(parents=True, exist_ok=True)
        dim_config = replace(config, dimension=dim, dimensions=[dim], experiment_dir=dim_dir)
        if config.mlp_units_from_dimension:
            dim_config = replace(dim_config, mlp_units=dim)
        if dim != 3:
            dim_config = replace(
                dim_config,
                gif_sphere=False,
                gif_histogram=False,
                pdf_trajectory=False,
            )
        run_experiment_s2(dim_config)
        dimension_dirs[dim] = dim_dir

    output_path = base_dir / "cluster_count_by_dimension"
    _save_dimension_cluster_count_plot(output_path, dimension_dirs, config.ascending)


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
    null_cluster_masses: list[list[float]],
    mlp_cluster_masses: list[list[float]],
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
    *,
    positions_initial: Optional[np.ndarray] = None,
    positions_middle_null: Optional[np.ndarray] = None,
    positions_middle_mlp: Optional[np.ndarray] = None,
    positions_final_null: Optional[np.ndarray] = None,
    positions_final_mlp: Optional[np.ndarray] = None,
    max_drift_final_null: Optional[list[float]] = None,
    max_drift_final_mlp: Optional[list[float]] = None,
    heaviest_mass_null: Optional[float] = None,
    heaviest_mass_mlp: Optional[float] = None,
    energy_times_null: Optional[list[float]] = None,
    energy_values_null: Optional[list[float]] = None,
    energy_times_mlp: Optional[list[float]] = None,
    energy_values_mlp: Optional[list[float]] = None,
    mlp_a: Optional[list[list[float]]] = None,
    mlp_omega: Optional[list[list[float]]] = None,
    mlp_activation: Optional[str] = None,
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
        "null_cluster_masses": null_cluster_masses,
        "mlp_cluster_masses": mlp_cluster_masses,
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
    # Add position snapshots if available (for regenerating images without re-simulation)
    if positions_initial is not None:
        summary["positions_initial"] = positions_initial.tolist()
    if positions_middle_null is not None:
        summary["positions_middle_null"] = positions_middle_null.tolist()
    if positions_middle_mlp is not None:
        summary["positions_middle_mlp"] = positions_middle_mlp.tolist()
    if positions_final_null is not None:
        summary["positions_final_null"] = positions_final_null.tolist()
    if positions_final_mlp is not None:
        summary["positions_final_mlp"] = positions_final_mlp.tolist()
    # Add max drift values at convergence
    if max_drift_final_null is not None:
        summary["max_drift_final_null"] = max_drift_final_null
    if max_drift_final_mlp is not None:
        summary["max_drift_final_mlp"] = max_drift_final_mlp
    # Add heaviest cluster mass
    if heaviest_mass_null is not None:
        summary["heaviest_mass_null"] = heaviest_mass_null
    if heaviest_mass_mlp is not None:
        summary["heaviest_mass_mlp"] = heaviest_mass_mlp
    # Add energy data
    if energy_times_null is not None and energy_values_null is not None:
        summary["energy_times_null"] = energy_times_null
        summary["energy_values_null"] = energy_values_null
    if energy_times_mlp is not None and energy_values_mlp is not None:
        summary["energy_times_mlp"] = energy_times_mlp
        summary["energy_values_mlp"] = energy_values_mlp
    # Add MLP params
    if mlp_a is not None:
        summary["mlp_a"] = mlp_a
    if mlp_omega is not None:
        summary["mlp_omega"] = mlp_omega
    if mlp_activation is not None:
        summary["mlp_activation"] = mlp_activation
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


def _load_latest_summaries_by_beta(experiment_dir: Path) -> dict[float, dict]:
    latest: dict[float, tuple[float, dict]] = {}
    if not experiment_dir.exists():
        return {}
    for summary_path in experiment_dir.rglob("summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        beta = data.get("beta")
        if beta is None:
            continue
        try:
            beta_value = float(beta)
        except (TypeError, ValueError):
            continue
        mtime = summary_path.stat().st_mtime
        current = latest.get(beta_value)
        if current is None or mtime > current[0]:
            latest[beta_value] = (mtime, data)
    return {beta: data for beta, (mtime, data) in latest.items()}


def _save_dimension_cluster_count_plot(
    output_path: Path,
    dimension_dirs: dict[int, Path],
    ascending: bool,
) -> None:
    series = {}
    for dim, dim_dir in dimension_dirs.items():
        summaries = _load_latest_summaries_by_beta(dim_dir)
        if not summaries:
            continue
        betas = np.array(sorted(summaries.keys()), dtype=float)
        sqrt_betas = np.sqrt(betas)
        mlp_means = []
        null_means = []
        for beta in betas:
            entry = summaries[beta]
            mlp_vals = np.asarray(entry.get("mlp_counts", []), dtype=float)
            null_vals = np.asarray(entry.get("null_counts", []), dtype=float)
            mlp_means.append(float(np.mean(mlp_vals)) if mlp_vals.size else np.nan)
            null_means.append(float(np.mean(null_vals)) if null_vals.size else np.nan)
        series[dim] = (sqrt_betas, np.asarray(mlp_means), np.asarray(null_means))

    if not series:
        return

    color_pool = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    dims_sorted = sorted(series.keys())
    if len(dims_sorted) > len(color_pool):
        raise ValueError(
            f"Not enough unique colors for dimensions: {len(dims_sorted)} > {len(color_pool)}"
        )

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def _make_plot(log_y: bool, suffix: str, include_null: bool) -> None:
        def _mask_values(values: np.ndarray) -> np.ndarray:
            mask = np.isfinite(values)
            if log_y:
                mask &= values > 0.0
            return mask

        def _build_layers(value_index: int) -> dict[int, tuple[int, int]]:
            groups: list[dict[str, object]] = []
            for dim in dims_sorted:
                x_vals, mlp_vals, null_vals = series[dim]
                y_vals = mlp_vals if value_index == 0 else null_vals
                mask = _mask_values(y_vals)
                if not np.any(mask):
                    continue
                x = x_vals[mask]
                y = y_vals[mask]
                matched = False
                for group in groups:
                    gx = group["x"]
                    gy = group["y"]
                    if (
                        isinstance(gx, np.ndarray)
                        and isinstance(gy, np.ndarray)
                        and x.shape == gx.shape
                        and np.allclose(x, gx, rtol=0.0, atol=1e-12)
                        and np.allclose(y, gy, rtol=0.0, atol=1e-12)
                    ):
                        group["dims"].append(dim)
                        matched = True
                        break
                if not matched:
                    groups.append({"x": x, "y": y, "dims": [dim]})
            layers: dict[int, tuple[int, int]] = {}
            for group in groups:
                dims = group["dims"]
                size = len(dims)
                for idx, dim in enumerate(dims):
                    layers[dim] = (idx, size)
            return layers

        mlp_layers = _build_layers(0)
        null_layers = _build_layers(1) if include_null and ascending else {}

        fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
        handles = []
        labels = []
        for idx, dim in enumerate(dims_sorted):
            color = color_pool[idx % len(color_pool)]
            x_vals, mlp_vals, null_vals = series[dim]
            mask = _mask_values(mlp_vals)
            if np.any(mask):
                layer_idx, layer_size = mlp_layers.get(dim, (0, 1))
                line_width = 1.6 + 0.5 * (layer_size - 1 - layer_idx)
                marker_size = 16 + 4 * (layer_size - 1 - layer_idx)
                ax.plot(
                    x_vals[mask],
                    mlp_vals[mask],
                    color=color,
                    linewidth=line_width,
                    linestyle="-",
                )
                ax.scatter(
                    x_vals[mask],
                    mlp_vals[mask],
                    facecolors=color,
                    edgecolors=color,
                    s=marker_size,
                    zorder=3,
                )
            if include_null and ascending:
                null_mask = _mask_values(null_vals)
                if np.any(null_mask):
                    layer_idx, layer_size = null_layers.get(dim, (0, 1))
                    line_width = 1.2 + 0.4 * (layer_size - 1 - layer_idx)
                    marker_size = 14 + 3 * (layer_size - 1 - layer_idx)
                    ax.plot(
                        x_vals[null_mask],
                        null_vals[null_mask],
                        color=color,
                        linewidth=line_width,
                        linestyle="--",
                    )
                    ax.scatter(
                        x_vals[null_mask],
                        null_vals[null_mask],
                        facecolors=color,
                        edgecolors=color,
                        s=marker_size,
                        zorder=3,
                    )
            handles.append(Line2D([0], [0], color=color, linewidth=2.0))
            labels.append(f"d={dim}")

        ax.set_xlabel(r"$\sqrt{\beta}$")
        ax.set_ylabel("cluster count")
        if log_y:
            positives = []
            for _, (_, mlp_vals, null_vals) in series.items():
                positives.append(mlp_vals[np.isfinite(mlp_vals) & (mlp_vals > 0.0)])
                if include_null and ascending:
                    positives.append(null_vals[np.isfinite(null_vals) & (null_vals > 0.0)])
            if positives:
                all_pos = np.concatenate(positives)
                if all_pos.size:
                    ax.set_ylim(bottom=float(np.min(all_pos)) * 0.8)
            ax.set_yscale("log")
            ax.grid(True, axis="y", which="both", linewidth=0.4, alpha=0.3)
        else:
            ax.set_ylim(bottom=0.0)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
        ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=8,
            loc="best",
            handlelength=1.4,
            labelspacing=0.3,
            borderaxespad=0.2,
        )

        save_figure(fig, output_path.with_name(f"{output_path.name}{suffix}"), formats=("pdf",))
        plt.close(fig)

    _make_plot(log_y=False, suffix="", include_null=True)
    _make_plot(log_y=True, suffix="_log", include_null=True)
    _make_plot(log_y=False, suffix="_mlp", include_null=False)
    _make_plot(log_y=True, suffix="_mlp_log", include_null=False)


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


_SNAPSHOT_FRACTIONS = {
    "init": 0.0,
    "q1": 0.25,
    "middle": 0.5,
    "q3": 0.75,
    "final": 1.0,
}
_SNAPSHOT_LABELS = ("init", "q1", "middle", "q3", "final")


def _snapshot_index(length: int, label: str) -> int:
    if length <= 0:
        return 0
    frac = _SNAPSHOT_FRACTIONS.get(label)
    if frac is None:
        raise ValueError(f"Unknown snapshot label: {label}")
    idx = int(round(frac * (length - 1)))
    return int(np.clip(idx, 0, length - 1))


def _snapshot_points(
    history: Optional[np.ndarray],
    label: str,
    fallback: Optional[np.ndarray],
    initial: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if label == "init" and initial is not None:
        return initial
    if history is None or history.size == 0:
        return fallback
    idx = _snapshot_index(len(history), label)
    return history[idx]


def _select_sparse_snapshots(
    sparse_snapshots: list[tuple[int, float, np.ndarray]],
    target_steps: list[int],
) -> list[tuple[float, np.ndarray]]:
    if not sparse_snapshots or not target_steps:
        return []
    steps = np.array([item[0] for item in sparse_snapshots], dtype=float)
    chosen: list[tuple[float, np.ndarray]] = []
    used: set[int] = set()
    for target in target_steps:
        idx = int(np.argmin(np.abs(steps - target)))
        if idx in used:
            continue
        used.add(idx)
        _, time_val, snapshot = sparse_snapshots[idx]
        chosen.append((time_val, snapshot))
    return chosen


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
    """Simulate until convergence.

    If save_history=False, keeps initial, quarter, middle, three-quarter, and final states.
    """
    x = x0.astype(np.float64, copy=True)
    times = [0.0]
    history = [x.copy()]
    counts = deque(maxlen=convergence_window)
    if progress is not None:
        if progress_every is None:
            progress_every = max(1, max_steps // 100)
        progress(0, max_steps)

    # For save_history=False, we track sparse snapshots to recover quarter/middle/three-quarter later.
    sparse_snapshots = []  # [(step, time, snapshot), ...]
    last_saved_step = 0

    for step in range(1, max_steps + 1):
        x = step_positions(x, sim_config, mlp_params)

        if step % sim_config.save_every == 0 or step == max_steps:
            if save_history:
                times.append(step * sim_config.dt)
                history.append(x.copy())
            elif step - last_saved_step >= sim_config.save_every * 10:
                # Save sparse snapshots periodically (every 10 save_every intervals)
                sparse_snapshots.append((step, step * sim_config.dt, x.copy()))
                last_saved_step = step
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
                )
                if mlp_params is not None:
                    drift_check += mlp_drift_vectors(x, mlp_params)
                drift_norm = np.linalg.norm(drift_check, axis=1)
                drift_ok = float(np.max(drift_norm)) <= drift_tol
            if len(counts) == convergence_window and len(set(counts)) == 1 and spread_ok and drift_ok:
                if progress is not None:
                    progress(step, max_steps)
                if not save_history:
                    target_steps = [
                        int(round(step * frac))
                        for frac in (0.25, 0.5, 0.75)
                        if 1 <= int(round(step * frac)) < step
                    ]
                    target_steps = sorted(set(target_steps))
                    for time_val, snap in _select_sparse_snapshots(sparse_snapshots, target_steps):
                        times.append(time_val)
                        history.append(snap)
                    times.append(step * sim_config.dt)
                    history.append(x.copy())
                return np.asarray(times), np.asarray(history), step, "convergence"

        if progress is not None and (step % progress_every == 0 or step == max_steps):
            progress(step, max_steps)

    if not save_history:
        target_steps = [
            int(round(max_steps * frac))
            for frac in (0.25, 0.5, 0.75)
            if 1 <= int(round(max_steps * frac)) < max_steps
        ]
        target_steps = sorted(set(target_steps))
        for time_val, snap in _select_sparse_snapshots(sparse_snapshots, target_steps):
            times.append(time_val)
            history.append(snap)
        times.append(max_steps * sim_config.dt)
        history.append(x.copy())
    return np.asarray(times), np.asarray(history), max_steps, "max_steps"


def run_experiment_s2(config: RunConfig) -> None:
    if len(config.dimensions) > 1:
        _run_multi_dim_experiment(config)
        return

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
    experiment_dir = _resolve_experiment_dir(config, experiment_stamp)

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
        # Cap the clustering threshold to avoid beta→0 collapsing everything into one cluster.
        # Dimension-dependent cap: pi/(2d).
        threshold = min(threshold, np.pi / (2.0 * config.dimension))
        match_null_to_mlp = (
            is_infinite
            and config.gradient_mlp
            and not config.ascending
            and config.dimension == 3
        )
        enable_plots = config.dimension == 3
        # For descending runs in higher dimensions, cap MLP-null steps.
        null_step_limit = 1000 if (not config.ascending and config.dimension > 3) else None
        null_num_steps = num_steps
        null_max_steps = config.max_steps
        if null_step_limit is not None:
            null_num_steps = min(null_num_steps, null_step_limit)
            null_max_steps = min(null_max_steps, null_step_limit)
        null_progress_every = progress_interval(null_num_steps)
        sim_config_null = (
            sim_config
            if null_num_steps == num_steps
            else replace(sim_config, num_steps=null_num_steps)
        )

        # Only keep first history for plots; count clusters immediately for others
        null_steps = []
        null_stop_reasons = []
        null_cluster_times: list[Optional[float]] = []
        null_counts = []
        null_cluster_masses: list[list[float]] = []
        null_did_not_converge = False
        example_initial = points0_list[0]
        example_null_final = None
        example_null_times = None
        example_null_hist = None
        need_full_history = enable_plots and (
            config.gif_sphere or config.gif_histogram or config.pdf_trajectory
        )

        if not match_null_to_mlp:
            for idx, points0 in enumerate(points0_list):
                # Only save full history for first run if needed for GIFs
                save_this_history = (idx == 0) and need_full_history
                label = f"beta={beta} MLP_null init {idx + 1}/{config.num_point_inits}"
                bar = ProgressHandle(null_num_steps, label=label)
                if is_infinite:
                    times, points_hist, step_count, stop_reason = _simulate_until_convergence_s2(
                        points0,
                        sim_config,
                        None,
                        threshold,
                        config.convergence_window,
                        null_max_steps,
                        config.convergence_drift_tol,
                        config.convergence_spread_factor,
                        progress=bar.update_to,
                        progress_every=null_progress_every,
                        save_history=save_this_history,
                    )
                    if stop_reason != "convergence":
                        null_did_not_converge = True
                else:
                    times, points_hist = simulate_positions(
                        points0,
                        sim_config_null,
                        None,
                        progress=bar.update_to,
                        progress_every=null_progress_every,
                        save_history=save_this_history,
                    )
                    step_count = null_num_steps
                    stop_reason = "fixed_time"
                bar.close()
                null_steps.append(step_count)
                null_stop_reasons.append(stop_reason)
                null_cluster_times.append(
                    step_count * config.dt if stop_reason == "convergence" else None
                )
                final_points = points_hist[-1]
                n_clusters = cluster_count_s2(final_points, threshold)
                masses = cluster_masses_s2(final_points, threshold)
                att_end = attention_drift_particles_vectors(
                    final_points, beta, config.attention_mode, self_attention=config.self_attention
                )
                max_drift = float(np.max(np.linalg.norm(att_end, axis=1))) if att_end.size else 0.0
                masses_str = ", ".join(f"{m:.3f}" for m in masses[:5])
                if len(masses) > 5:
                    masses_str += ", ..."
                print(
                    f"  [null init {idx + 1}] stop_reason={stop_reason}, "
                    f"clusters={n_clusters}, masses=[{masses_str}], max|drift|={max_drift:.6e}"
                )

                # Count clusters immediately
                if is_infinite:
                    conv_idx = len(points_hist) - 1
                else:
                    conv_idx = convergence_index_s2(points_hist, threshold, config.convergence_window)
                null_counts.append(cluster_count_s2(points_hist[conv_idx], threshold))
                masses_conv = cluster_masses_s2(points_hist[conv_idx], threshold)
                null_cluster_masses.append(masses_conv.tolist())

                # Only keep first history for example plots
                if idx == 0:
                    example_null_final = points_hist[-1]
                    example_null_times = times
                    # For save_history=False, points_hist has [initial, q1, middle, q3, final],
                    # which is small enough to keep for snapshot extraction.
                    example_null_hist = points_hist if (need_full_history or (is_infinite and len(points_hist) <= 6)) else None
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
            mlp_cluster_masses: list[list[float]] = []
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
                    final_points = points_hist[-1]
                    n_clusters = cluster_count_s2(final_points, threshold)
                    masses = cluster_masses_s2(final_points, threshold)
                    att_end = attention_drift_particles_vectors(
                        final_points, beta, config.attention_mode, self_attention=config.self_attention
                    )
                    mlp_end = mlp_drift_vectors(final_points, mlp_params)
                    total_end = att_end + mlp_end
                    max_drift = float(np.max(np.linalg.norm(total_end, axis=1))) if total_end.size else 0.0
                    masses_str = ", ".join(f"{m:.3f}" for m in masses[:5])
                    if len(masses) > 5:
                        masses_str += ", ..."
                    print(
                        f"  [MLP {i + 1} init {j + 1}] stop_reason={stop_reason}, "
                        f"clusters={n_clusters}, masses=[{masses_str}], max|drift|={max_drift:.6e}"
                    )
                    
                    # Count clusters immediately
                    if is_infinite:
                        conv_idx = len(points_hist) - 1
                    else:
                        conv_idx = convergence_index_s2(points_hist, threshold, config.convergence_window)
                    mlp_counts.append(cluster_count_s2(points_hist[conv_idx], threshold))
                    masses_conv = cluster_masses_s2(points_hist[conv_idx], threshold)
                    mlp_cluster_masses.append(masses_conv.tolist())
                    
                    # Only keep first history for example plots
                    if i == 0 and j == 0:
                        example_mlp_final = points_hist[-1]
                        example_mlp_times = times
                        # For save_history=False, points_hist has [initial, q1, middle, q3, final],
                        # which is small enough to keep for snapshot extraction.
                        example_mlp_hist = points_hist if (need_full_history or (is_infinite and len(points_hist) <= 6)) else None
                    # Discard history for non-first runs to save memory
                    del points_hist, times

            if match_null_to_mlp:
                run_null_steps = []
                run_null_stop_reasons = []
                run_null_cluster_times: list[Optional[float]] = []
                run_null_counts = []
                run_null_cluster_masses: list[list[float]] = []
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
                    final_points = points_hist[-1]
                    n_clusters = cluster_count_s2(final_points, threshold)
                    masses = cluster_masses_s2(final_points, threshold)
                    att_end = attention_drift_particles_vectors(
                        final_points, beta, config.attention_mode, self_attention=config.self_attention
                    )
                    max_drift = float(np.max(np.linalg.norm(att_end, axis=1))) if att_end.size else 0.0
                    masses_str = ", ".join(f"{m:.3f}" for m in masses[:5])
                    if len(masses) > 5:
                        masses_str += ", ..."
                    print(
                        f"  [null init {idx + 1}] stop_reason={stop_reason}, "
                        f"clusters={n_clusters}, masses=[{masses_str}], max|drift|={max_drift:.6e}"
                    )

                    # Count clusters immediately
                    conv_idx = len(points_hist) - 1
                    run_null_counts.append(cluster_count_s2(points_hist[conv_idx], threshold))
                    masses_conv = cluster_masses_s2(points_hist[conv_idx], threshold)
                    run_null_cluster_masses.append(masses_conv.tolist())

                    # Only keep first history for example plots
                    if idx == 0:
                        example_null_final = points_hist[-1]
                        example_null_times = times
                        # When save_history=False, points_hist has [initial, q1, middle, q3, final]
                        # which is small enough to keep for snapshot extraction.
                        example_null_hist = points_hist if (need_full_history or len(points_hist) <= 6) else None
                    # Discard history for non-first runs to save memory
                    del points_hist, times

                all_steps.extend(run_null_steps)
            else:
                run_null_steps = null_steps
                run_null_stop_reasons = null_stop_reasons
                run_null_cluster_times = null_cluster_times
                run_null_counts = null_counts
                run_null_cluster_masses = null_cluster_masses
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
            # Add MLP a and omega arrays if we have mlp_params
            if mlp_params_list:
                first_mlp = mlp_params_list[0]
                params["mlp_a"] = first_mlp.a.tolist()
                params["mlp_omega"] = first_mlp.omega.tolist()
            write_json(run_dir / "params.json", params, compact=False)
            run_seconds = time.perf_counter() - run_start
            
            # Compute middle positions for summary (before generating images)
            mid_null = _snapshot_points(example_null_hist, "middle", example_null_final, example_initial)
            mid_mlp = _snapshot_points(example_mlp_hist, "middle", example_mlp_final, example_initial)
            if mid_mlp is None:
                mid_mlp = mid_null
            
            # Compute max drift at convergence for null model
            max_drift_null_list = []
            if example_null_final is not None:
                null_drift = attention_drift_particles_vectors(
                    example_null_final,
                    beta,
                    sim_config.attention_mode,
                    self_attention=sim_config.self_attention,
                )
                max_drift_null = float(np.max(np.linalg.norm(null_drift, axis=1)))
                max_drift_null_list = [max_drift_null] if config.num_point_inits > 0 else []
            
            # Compute max drift at convergence for MLP model
            max_drift_mlp_list = []
            if example_mlp_final is not None and example_mlp_params is not None:
                mlp_drift = mlp_drift_vectors(example_mlp_final, example_mlp_params)
                att_drift = attention_drift_particles_vectors(
                    example_mlp_final,
                    beta,
                    sim_config.attention_mode,
                    self_attention=sim_config.self_attention,
                )
                total_drift_mlp = att_drift + mlp_drift
                max_drift_mlp = float(np.max(np.linalg.norm(total_drift_mlp, axis=1)))
                max_drift_mlp_list = [max_drift_mlp] if config.num_mlp_inits > 0 else []
            
            # Compute heaviest cluster mass at final time
            heaviest_null = None
            heaviest_mlp = None
            if example_null_final is not None:
                heaviest_null = heaviest_cluster_mass_s2(example_null_final, threshold)
            if example_mlp_final is not None:
                heaviest_mlp = heaviest_cluster_mass_s2(example_mlp_final, threshold)
            
            # Compute energy time series (for summary storage)
            # If full history exists, compute full energy curve; otherwise compute initial/final only
            energy_times_null = None
            energy_values_null = None
            energy_times_mlp = None
            energy_values_mlp = None
            if example_null_hist is not None and example_null_times is not None and len(example_null_hist) > 0:
                energy_times_null = [float(t) for t in example_null_times]
                energy_values_null = [float(compute_total_energy(pts, beta, None)) for pts in example_null_hist]
            elif example_initial is not None and example_null_final is not None:
                # Compute only initial and final energy
                energy_times_null = [0.0, run_null_cluster_times[0] if run_null_cluster_times else 0.0]
                energy_values_null = [
                    float(compute_total_energy(example_initial, beta, None)),
                    float(compute_total_energy(example_null_final, beta, None)),
                ]
            if example_mlp_hist is not None and example_mlp_times is not None and len(example_mlp_hist) > 0:
                energy_times_mlp = [float(t) for t in example_mlp_times]
                energy_values_mlp = [float(compute_total_energy(pts, beta, example_mlp_params)) for pts in example_mlp_hist]
            elif example_initial is not None and example_mlp_final is not None:
                # Compute only initial and final energy
                energy_times_mlp = [0.0, mlp_cluster_times[0] if mlp_cluster_times else 0.0]
                energy_values_mlp = [
                    float(compute_total_energy(example_initial, beta, example_mlp_params)),
                    float(compute_total_energy(example_mlp_final, beta, example_mlp_params)),
                ]
            
            # Extract MLP params for summary
            mlp_a_list = None
            mlp_omega_list = None
            mlp_activation_str = None
            if example_mlp_params is not None:
                mlp_a_list = example_mlp_params.a.tolist()
                mlp_omega_list = example_mlp_params.omega.tolist()
                mlp_activation_str = example_mlp_params.activation
            
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
                run_null_cluster_masses,
                mlp_cluster_masses,
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
                positions_initial=example_initial,
                positions_middle_null=mid_null,
                positions_middle_mlp=mid_mlp,
                positions_final_null=example_null_final,
                positions_final_mlp=example_mlp_final,
                max_drift_final_null=max_drift_null_list if max_drift_null_list else None,
                max_drift_final_mlp=max_drift_mlp_list if max_drift_mlp_list else None,
                heaviest_mass_null=heaviest_null,
                heaviest_mass_mlp=heaviest_mlp,
                energy_times_null=energy_times_null,
                energy_values_null=energy_values_null,
                energy_times_mlp=energy_times_mlp,
                energy_values_mlp=energy_values_mlp,
                mlp_a=mlp_a_list,
                mlp_omega=mlp_omega_list,
                mlp_activation=mlp_activation_str,
            )

            if enable_plots and example_mlp_final is not None:
                potential_params = None
                if config.gradient_mlp and example_mlp_params is not None:
                    potential_params = (
                        example_mlp_params.a,
                        example_mlp_params.omega,
                        example_mlp_params.activation,
                    )
                # Create subdirectories
                sphere_dir = run_dir / "sphere"
                hist_dir = run_dir / "hist"
                sphere_dir.mkdir(exist_ok=True)
                hist_dir.mkdir(exist_ok=True)
                
                import matplotlib.pyplot as plt

                if potential_params is not None:
                    fig = make_mlp_potential_surface_figure(*potential_params)
                    save_figure(fig, sphere_dir / "mlp_potential", formats=("pdf",))
                    plt.close(fig)
                
                # ---------- SPHERE figures at init, q1, middle, q3, final ----------
                for label in _SNAPSHOT_LABELS:
                    null_points = _snapshot_points(
                        example_null_hist,
                        label,
                        example_null_final,
                        example_initial,
                    )
                    mlp_points = _snapshot_points(
                        example_mlp_hist,
                        label,
                        example_mlp_final,
                        example_initial,
                    )
                    if null_points is not None:
                        fig = make_s2_single_figure(
                            null_points,
                            NULL_COLOR,
                            potential_params=None,
                            show_potential=False,
                            point_size=10.0,
                        )
                        save_figure(fig, sphere_dir / f"{label}_null", formats=("pdf",))
                        plt.close(fig)
                    if mlp_points is not None:
                        fig = make_s2_single_figure(
                            mlp_points,
                            mlp_color,
                            potential_params=potential_params,
                            show_potential=config.gradient_mlp,
                            point_size=10.0,
                        )
                        save_figure(fig, sphere_dir / f"{label}_mlp", formats=("pdf",))
                        plt.close(fig)

                # ---------- HISTOGRAMS at init, q1, middle, q3, final ----------
                for label in _SNAPSHOT_LABELS:
                    null_points = _snapshot_points(
                        example_null_hist,
                        label,
                        example_null_final,
                        example_initial,
                    )
                    mlp_points = _snapshot_points(
                        example_mlp_hist,
                        label,
                        example_mlp_final,
                        example_initial,
                    )
                    if null_points is not None:
                        fig = make_s2_histogram_bar_figure(
                            null_points,
                            NULL_COLOR,
                            potential_params=None,
                            show_potential=False,
                        )
                        save_figure(fig, hist_dir / f"{label}_null", formats=("pdf",))
                        plt.close(fig)
                    if mlp_points is not None:
                        fig = make_s2_histogram_bar_figure(
                            mlp_points,
                            mlp_color,
                            potential_params=potential_params,
                            show_potential=config.gradient_mlp,
                            show_decision_boundaries=False,
                        )
                        save_figure(fig, hist_dir / f"{label}_mlp", formats=("pdf",))
                        plt.close(fig)
                # With decision boundaries
                fig = make_s2_histogram_bar_figure(
                    example_mlp_final,
                    mlp_color,
                    potential_params=potential_params,
                    show_potential=config.gradient_mlp,
                    show_decision_boundaries=True,
                )
                save_figure(fig, hist_dir / "final_mlp_boundaries", formats=("pdf",))
                plt.close(fig)
                if config.pdf_trajectory and example_null_hist is not None and example_null_times is not None:
                    # Linear scale
                    fig = make_s2_trajectory_figure(
                        example_null_hist,
                        example_null_times,
                        NULL_COLOR,
                        time_scale="linear",
                    )
                    save_figure(fig, sphere_dir / "trajectory_null", formats=("pdf",))
                    plt.close(fig)
                    # Log scale
                    fig = make_s2_trajectory_figure(
                        example_null_hist,
                        example_null_times,
                        NULL_COLOR,
                        time_scale="log",
                    )
                    save_figure(fig, sphere_dir / "trajectory_null_log", formats=("pdf",))
                    plt.close(fig)
                if config.pdf_trajectory and example_mlp_hist is not None and example_mlp_times is not None:
                    # Linear scale
                    fig = make_s2_trajectory_figure(
                        example_mlp_hist,
                        example_mlp_times,
                        mlp_color,
                        time_scale="linear",
                    )
                    save_figure(fig, sphere_dir / "trajectory_mlp", formats=("pdf",))
                    plt.close(fig)
                    # Log scale
                    fig = make_s2_trajectory_figure(
                        example_mlp_hist,
                        example_mlp_times,
                        mlp_color,
                        time_scale="log",
                    )
                    save_figure(fig, sphere_dir / "trajectory_mlp_log", formats=("pdf",))
                    plt.close(fig)
                
                # ---------- ENERGY plots ----------
                if example_null_hist is not None and example_mlp_hist is not None:
                    # Compute energy for each time step
                    energy_null = np.array([
                        compute_total_energy(pts, beta, None)
                        for pts in example_null_hist
                    ])
                    energy_mlp = np.array([
                        compute_total_energy(pts, beta, example_mlp_params)
                        for pts in example_mlp_hist
                    ])
                    # Linear scale
                    fig = make_energy_figure(
                        example_null_times,
                        energy_null,
                        example_mlp_times,
                        energy_mlp,
                        NULL_COLOR,
                        mlp_color,
                        time_scale="linear",
                    )
                    save_figure(fig, run_dir / "energy", formats=("pdf",))
                    plt.close(fig)
                    # Log scale
                    fig = make_energy_figure(
                        example_null_times,
                        energy_null,
                        example_mlp_times,
                        energy_mlp,
                        NULL_COLOR,
                        mlp_color,
                        time_scale="log",
                    )
                    save_figure(fig, run_dir / "energy_log", formats=("pdf",))
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
        betas = np.asarray([float(entry["beta"]) for entry in summaries])
        order = np.argsort(sqrt_betas)
        sqrt_betas = sqrt_betas[order]
        betas_sorted = betas[order]
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
        
        # ---------- Heaviest cluster mass plot ----------
        heaviest_null_list = []
        heaviest_mlp_list = []
        smallest_mlp_list = []
        mlp_a = None
        mlp_omega = None
        mlp_activation = None

        def _smallest_non_spurious(masses, params_json: Optional[str]) -> float:
            if masses is None:
                return float("nan")
            try:
                masses_arr = np.asarray(masses, dtype=float)
            except Exception:
                return float("nan")
            if masses_arr.ndim > 1:
                masses_arr = masses_arr[0]
            if masses_arr.size == 0:
                return float("nan")
            mass_threshold = 0.0
            n_particles = 0
            if params_json:
                try:
                    params = json.loads(params_json)
                    mass_threshold = float(params.get("mass_threshold", 0.0))
                    n_particles = int(params.get("n_particles", 0))
                except Exception:
                    pass
            min_mass = mass_threshold
            if n_particles > 0:
                min_mass = max(min_mass, 1.0 / n_particles)
            valid = masses_arr[masses_arr >= min_mass]
            if valid.size == 0:
                return float("nan")
            return float(np.min(valid))
        
        for idx in order:
            entry = summaries[idx]
            h_null = entry.get("heaviest_mass_null")
            h_mlp = entry.get("heaviest_mass_mlp")
            if h_null is None:
                positions = entry.get("positions_final_null")
                params_json = entry.get("params_json")
                if positions is not None and params_json:
                    try:
                        params = json.loads(params_json)
                        cluster_scale = params.get("cluster_scale")
                        beta_val = float(entry["beta"])
                        if cluster_scale is not None and beta_val > 0.0:
                            threshold = cluster_threshold(beta_val, float(cluster_scale))
                            h_null = heaviest_cluster_mass_s2(
                                np.asarray(positions, dtype=np.float64),
                                threshold,
                            )
                    except Exception:
                        h_null = None
            if h_mlp is None:
                positions = entry.get("positions_final_mlp")
                params_json = entry.get("params_json")
                if positions is not None and params_json:
                    try:
                        params = json.loads(params_json)
                        cluster_scale = params.get("cluster_scale")
                        beta_val = float(entry["beta"])
                        if cluster_scale is not None and beta_val > 0.0:
                            threshold = cluster_threshold(beta_val, float(cluster_scale))
                            h_mlp = heaviest_cluster_mass_s2(
                                np.asarray(positions, dtype=np.float64),
                                threshold,
                            )
                    except Exception:
                        h_mlp = None
            heaviest_null_list.append(h_null if h_null is not None else np.nan)
            heaviest_mlp_list.append(h_mlp if h_mlp is not None else np.nan)
            masses_list = entry.get("mlp_cluster_masses")
            masses = None
            if isinstance(masses_list, list) and masses_list:
                masses = masses_list[0]
            smallest_mlp_list.append(_smallest_non_spurious(masses, entry.get("params_json")))

            if mlp_a is None and mlp_omega is None and mlp_activation is None:
                mlp_a_raw = entry.get("mlp_a")
                mlp_omega_raw = entry.get("mlp_omega")
                mlp_activation = entry.get("mlp_activation")
                if mlp_a_raw is not None and mlp_omega_raw is not None:
                    mlp_a = np.array(mlp_a_raw)
                    mlp_omega = np.array(mlp_omega_raw)
            
            # Get MLP params from params.json (via summary's run_dir)
            if mlp_a is None and mlp_omega is None and mlp_activation is None:
                run_dir_str = entry.get("run_dir")
                if run_dir_str:
                    params_path = Path(run_dir_str) / "params.json"
                    if params_path.exists():
                        try:
                            params_data = json.loads(params_path.read_text(encoding="utf-8"))
                            mlp_a_raw = params_data.get("mlp_a")
                            mlp_omega_raw = params_data.get("mlp_omega")
                            mlp_activation = params_data.get("activation")
                            if mlp_a_raw is not None and mlp_omega_raw is not None:
                                mlp_a = np.array(mlp_a_raw)
                                mlp_omega = np.array(mlp_omega_raw)
                        except Exception:
                            pass
        
        heaviest_null_arr = np.array(heaviest_null_list)
        heaviest_mlp_arr = np.array(heaviest_mlp_list)
        smallest_mlp_arr = np.array(smallest_mlp_list)
        
        fig = make_heaviest_mass_figure(
            betas_sorted,
            heaviest_null_arr,
            heaviest_mlp_arr,
            smallest_mlp=smallest_mlp_arr,
            mlp_a=mlp_a,
            mlp_omega=mlp_omega,
            mlp_activation=mlp_activation,
            null_color=NULL_COLOR,
            mlp_color=MLP_COLOR,
        )
        save_figure(
            fig,
            stats_dir / "heaviest_mass",
            formats=("pdf", "png"),
            dpi_by_format={"pdf": 2400},
        )
        plt.close(fig)

        # ---------- All cluster masses plot ----------
        all_masses_list = []
        for idx in order:
            entry = summaries[idx]
            masses_list = entry.get("mlp_cluster_masses")
            masses_flat: list[float] = []
            if isinstance(masses_list, list):
                for masses in masses_list:
                    try:
                        masses_flat.extend(float(m) for m in masses)
                    except Exception:
                        continue
            if not masses_flat:
                positions = entry.get("positions_final_mlp")
                params_json = entry.get("params_json")
                if positions is not None and params_json:
                    try:
                        params = json.loads(params_json)
                        cluster_scale = params.get("cluster_scale")
                        beta_val = float(entry["beta"])
                        if cluster_scale is not None and beta_val > 0.0:
                            threshold = cluster_threshold(beta_val, float(cluster_scale))
                            masses_flat = cluster_masses_s2(
                                np.asarray(positions, dtype=np.float64),
                                threshold,
                            ).tolist()
                    except Exception:
                        masses_flat = []
            all_masses_list.append(masses_flat)

        fig = make_all_masses_figure(
            betas_sorted,
            all_masses_list,
            mlp_a=mlp_a,
            mlp_omega=mlp_omega,
            mlp_activation=mlp_activation,
            mlp_color=MLP_COLOR,
        )
        save_figure(
            fig,
            stats_dir / "all_masses",
            formats=("pdf", "png"),
            dpi_by_format={"pdf": 2400},
        )
        plt.close(fig)
        


def main(config_path: Optional[Path] = None) -> None:
    path = config_path or Path("config.json")
    config = load_config(path)
    run_experiment_s2(config)


if __name__ == "__main__":
    main()
