"""Simulation runner for S1 self-attention + MLP drift simulations."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import time
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
from tqdm import tqdm

from .analysis import (
    cluster_count,
    cluster_max_spread,
    cluster_threshold,
    convergence_index,
    mass_count,
    mode_count,
)
from .config import RunConfig, SeedPlan, build_seed_plan, load_config
from .dynamics import (
    MLPConfig,
    MLPParams,
    SimulationConfig,
    TWO_PI,
    attention_drift_at,
    attention_drift_particles,
    mlp_drift,
    sample_mlp_params,
    sample_theta0,
    simulate,
    step_theta,
)
from .io import canonical_json, find_matching_run, make_run_dir, save_gif_from_images, write_json
from .plotting import (
    MLP_COLOR,
    NULL_COLOR,
    SA_COLOR,
    gamma_k_s1,
    make_cluster_bar_plot,
    make_cluster_bar_plot_with_null,
    make_field_frame,
    make_gamma_figure,
    make_histogram_comparison_frame,
    make_histogram_figure,
    make_histogram_frame,
    make_mlp_scale_stop_time_figure,
    make_total_clusters_figure,
    make_trajectory_figure,
    mlp_potential,
    save_figure,
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
        self._bar = tqdm(total=self.total, desc=self.label, unit=self.unit, leave=False)

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
    return tqdm(iterable, desc=label, unit=unit, leave=False)


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

    return {
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
        "gifs": config.gifs,
    }


def _frame_indices(times: np.ndarray, interval: float) -> tuple[list[int], list[float]]:
    if times.size == 0:
        return [], []
    target_times = np.arange(0.0, times[-1] + 1e-9, interval)
    indices = np.searchsorted(times, target_times, side="left")
    indices = np.clip(indices, 0, len(times) - 1)
    unique_indices = []
    unique_times = []
    seen = set()
    for idx in indices.tolist():
        if idx in seen:
            continue
        seen.add(idx)
        unique_indices.append(idx)
        unique_times.append(float(times[idx]))
    return unique_indices, unique_times


def _histogram_density(theta: np.ndarray, edges: np.ndarray) -> list[float]:
    counts, _ = np.histogram(theta, bins=edges, density=True)
    return counts.astype(float).tolist()


def _stop_time(cluster_times: list[Optional[float]]) -> float:
    if not cluster_times:
        return 0.0
    values = [0.0 if t is None else float(t) for t in cluster_times]
    return float(np.mean(values))


def _simulate_until_convergence(
    theta0: np.ndarray,
    sim_config: SimulationConfig,
    mlp_params,
    threshold: float,
    convergence_window: int,
    max_steps: int,
    drift_tol: float,
    spread_factor: float,
    progress: Optional[Callable[[int, int], None]] = None,
    progress_every: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    theta = theta0.astype(np.float64, copy=True)
    times = [0.0]
    history = [theta.copy()]
    counts = deque(maxlen=convergence_window)

    if progress is not None:
        if progress_every is None:
            progress_every = max(1, max_steps // 100)
        progress(0, max_steps)

    for step in range(1, max_steps + 1):
        theta = step_theta(theta, sim_config, mlp_params)

        if step % sim_config.save_every == 0 or step == max_steps:
            times.append(step * sim_config.dt)
            history.append(theta.copy())
            counts.append(cluster_count(theta, threshold))
            max_spread = cluster_max_spread(theta, threshold)
            spread_ok = spread_factor <= 0.0 or max_spread <= spread_factor * threshold
            if drift_tol <= 0.0:
                drift_ok = True
            else:
                drift_check = attention_drift_particles(
                    theta,
                    sim_config.beta,
                    sim_config.attention_mode,
                    self_attention=sim_config.self_attention,
                    ascending=sim_config.ascending,
                )
                if mlp_params is not None:
                    drift_check += mlp_drift(theta, mlp_params)
                drift_ok = np.max(np.abs(drift_check)) <= drift_tol
            if len(counts) == convergence_window and len(set(counts)) == 1 and spread_ok and drift_ok:
                if progress is not None:
                    progress(step, max_steps)
                return np.asarray(times), np.asarray(history), step, True

        if progress is not None and (step % progress_every == 0 or step == max_steps):
            progress(step, max_steps)

    return np.asarray(times), np.asarray(history), max_steps, False


def _save_frames_and_gif(
    run_dir,
    label_slug: str,
    theta_hist: np.ndarray,
    times: np.ndarray,
    beta: float,
    n_particles: int,
    mlp_title: str,
    attention_label: str,
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    color: str,
    plot_interval: float,
    output_frame_limit: int,
    show_potential: bool = True,
    save_gif: bool = True,
) -> None:
    frame_indices, frame_times = _frame_indices(times, plot_interval)
    if not frame_indices:
        return

    skip_gif = len(frame_indices) > output_frame_limit or not save_gif
    frames_dir = run_dir / f"frames_{label_slug}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    first_idx = 0
    last_idx = len(frame_indices) - 1
    mid_idx = len(frame_indices) // 2
    frame_positions = [first_idx, mid_idx, last_idx]
    frame_positions = [p for p in frame_positions if 0 <= p < len(frame_indices)]
    seen = []
    for p in frame_positions:
        if p not in seen:
            seen.append(p)
    frame_positions = seen
    frame_labels = {}
    for pos in frame_positions:
        if pos == first_idx:
            frame_labels[pos] = "first"
        elif pos == last_idx:
            frame_labels[pos] = "last"
        else:
            frame_labels[pos] = "middle"

    if skip_gif:
        frame_iter = frame_positions
    else:
        frame_iter = iter_progress(range(len(frame_indices)), label=f"Frames {label_slug}", unit="frame")
    gif_images = []

    if not skip_gif:
        import io
        from PIL import Image

    for frame_pos in frame_iter:
        if skip_gif:
            time_idx = frame_indices[frame_pos]
            frame_time = frame_times[frame_pos]
        else:
            time_idx = frame_indices[frame_pos]
            frame_time = frame_times[frame_pos]
        shade_regions = True
        frame_fig = make_histogram_frame(
            theta_hist[time_idx],
            frame_time,
            beta,
            n_particles,
            mlp_title,
            attention_label,
            a=a,
            omega=omega,
            activation=activation,
            color=color,
            shade_regions=shade_regions,
            show_potential=show_potential,
        )
        if not skip_gif:
            buf = io.BytesIO()
            frame_fig.savefig(buf, format="png", dpi=120)
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            gif_images.append(img.copy())
            img.close()
            buf.close()
        import matplotlib.pyplot as plt

        if frame_pos in frame_labels:
            label = frame_labels[frame_pos]
            if frame_fig.axes:
                for ax in frame_fig.axes:
                    ax.set_title("")
            frame_path = frames_dir / f"frame_{label}.pdf"
            frame_fig.savefig(frame_path, dpi=150)

            if a.size:
                alt_shade = not shade_regions
                alt_fig = make_histogram_frame(
                    theta_hist[time_idx],
                    frame_time,
                    beta,
                    n_particles,
                    mlp_title,
                    attention_label,
                    a=a,
                    omega=omega,
                    activation=activation,
                    color=color,
                    shade_regions=alt_shade,
                    show_potential=show_potential,
                )
                if alt_fig.axes:
                    for ax in alt_fig.axes:
                        ax.set_title("")
                suffix = "shaded" if alt_shade else "noshade"
                alt_path = frames_dir / f"frame_{label}_{suffix}.pdf"
                alt_fig.savefig(alt_path, dpi=150)
                plt.close(alt_fig)

        plt.close(frame_fig)

    if not save_gif:
        return
    if skip_gif:
        print(
            f"  Skipping GIF for {label_slug}: {len(frame_indices)} frames exceeds "
            f"output_frame_limit={output_frame_limit}."
        )
        return

    gif_path = run_dir / f"evolution_{label_slug}.gif"
    if not save_gif_from_images(gif_images, gif_path, plot_interval):
        print(f"Warning: GIF not generated for {label_slug}.")
    for img in gif_images:
        try:
            img.close()
        except Exception:
            pass


def _save_field_gif(
    run_dir,
    label_slug: str,
    theta_hist: np.ndarray,
    times: np.ndarray,
    beta: float,
    n_particles: int,
    mlp_title: str,
    attention_label: str,
    attention_mode: str,
    ascending: bool,
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    plot_interval: float,
    output_frame_limit: int,
    show_potential: bool = True,
    grid_points: int = 256,
) -> None:
    frame_indices, frame_times = _frame_indices(times, plot_interval)
    if not frame_indices:
        return
    if len(frame_indices) > output_frame_limit:
        print(
            f"  Skipping field GIF for {label_slug}: {len(frame_indices)} frames exceeds "
            f"output_frame_limit={output_frame_limit}."
        )
        return

    theta_grid = np.linspace(0.0, TWO_PI, grid_points, endpoint=False)
    params = MLPParams(a=a, omega=omega, activation=activation)
    potential = None
    if show_potential:
        potential = mlp_potential(theta_grid, a, omega, activation)

    fields = []
    att_fields = []
    mlp_fields = []
    max_abs = 0.0
    for idx in frame_indices:
        theta_particles = theta_hist[idx]
        att = attention_drift_at(
            theta_grid,
            theta_particles,
            beta,
            attention_mode,
            ascending=ascending,
        )
        mlp = mlp_drift(theta_grid, params)
        field = att + mlp
        fields.append(field)
        att_fields.append(att)
        mlp_fields.append(mlp)
        max_abs = max(
            max_abs,
            float(np.max(np.abs(field))),
            float(np.max(np.abs(att))),
            float(np.max(np.abs(mlp))),
        )

    if max_abs <= 0.0:
        max_abs = 1.0
    y_limits = (-max_abs, max_abs)

    gif_images = []
    import io
    from PIL import Image

    for field, att, mlp, t in zip(fields, att_fields, mlp_fields, frame_times):
        fig = make_field_frame(
            theta_grid,
            field,
            att,
            mlp,
            potential,
            t,
            beta,
            n_particles,
            mlp_title,
            attention_label,
            y_limits,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        gif_images.append(img.copy())
        img.close()
        buf.close()
        import matplotlib.pyplot as plt

        plt.close(fig)

    gif_path = run_dir / f"field_{label_slug}.gif"
    if not save_gif_from_images(gif_images, gif_path, plot_interval):
        print(f"Warning: field GIF not generated for {label_slug}.")
    for img in gif_images:
        try:
            img.close()
        except Exception:
            pass


def _common_frame_indices(
    times_a: np.ndarray,
    times_b: np.ndarray,
    interval: float,
) -> tuple[list[int], list[int], list[float]]:
    if times_a.size == 0 or times_b.size == 0:
        return [], [], []
    max_time = max(float(times_a[-1]), float(times_b[-1]))
    if max_time <= 0.0:
        target_times = np.array([0.0])
    else:
        target_times = np.arange(0.0, max_time + 1e-9, interval)
    idx_a = np.searchsorted(times_a, target_times, side="left")
    idx_b = np.searchsorted(times_b, target_times, side="left")
    idx_a = np.clip(idx_a, 0, len(times_a) - 1)
    idx_b = np.clip(idx_b, 0, len(times_b) - 1)

    filtered_a: list[int] = []
    filtered_b: list[int] = []
    filtered_times: list[float] = []
    prev_key: Optional[tuple[int, int]] = None
    for a_idx, b_idx, t in zip(idx_a, idx_b, target_times):
        key = (int(a_idx), int(b_idx))
        if key == prev_key:
            continue
        prev_key = key
        filtered_a.append(key[0])
        filtered_b.append(key[1])
        filtered_times.append(float(t))
    return filtered_a, filtered_b, filtered_times


def _save_comparison_gif(
    output_path: Path,
    null_hist: np.ndarray,
    null_times: np.ndarray,
    mlp_hist: np.ndarray,
    mlp_times: np.ndarray,
    beta: float,
    n_particles: int,
    mlp_std_label: str,
    attention_label: str,
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    plot_interval: float,
    output_frame_limit: int,
    show_potential: bool = True,
    mlp_color: str = MLP_COLOR,
) -> None:
    null_indices, mlp_indices, frame_times = _common_frame_indices(
        null_times, mlp_times, plot_interval
    )
    if not frame_times:
        return
    if len(frame_times) > output_frame_limit:
        print(
            f"  Skipping GIF for {output_path.name}: {len(frame_times)} frames exceeds "
            f"output_frame_limit={output_frame_limit}."
        )
        return

    frame_iter = iter_progress(range(len(frame_times)), label=f"Frames {output_path.stem}", unit="frame")
    gif_images = []

    import io
    from PIL import Image

    for frame_idx in frame_iter:
        t = frame_times[frame_idx]
        null_idx = null_indices[frame_idx]
        mlp_idx = mlp_indices[frame_idx]
        fig = make_histogram_comparison_frame(
            null_hist[null_idx],
            mlp_hist[mlp_idx],
            t,
            beta,
            n_particles,
            mlp_std_label,
            attention_label,
            a=a,
            omega=omega,
            activation=activation,
            show_potential=show_potential,
            mlp_color=mlp_color,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        gif_images.append(img.copy())
        img.close()
        buf.close()
        import matplotlib.pyplot as plt

        plt.close(fig)

    if not save_gif_from_images(gif_images, output_path, plot_interval):
        print(f"Warning: GIF not generated for {output_path.name}.")
    for img in gif_images:
        try:
            img.close()
        except Exception:
            pass



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


def run_experiment(config: RunConfig) -> None:
    is_infinite = np.isinf(config.total_time)
    if is_infinite:
        num_steps = int(config.max_steps)
    else:
        num_steps = int(round(config.total_time / config.dt))
    print("Run configuration:")
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
    print(f"  plot_interval={config.plot_interval}")
    print(f"  cluster_scale={config.cluster_scale}")
    print(f"  mass_threshold={config.mass_threshold}")
    print(f"  convergence_window={config.convergence_window}")
    print(f"  convergence_drift_tol={config.convergence_drift_tol}")
    print(f"  convergence_spread_factor={config.convergence_spread_factor}")
    print(f"  self_attention={config.self_attention}")
    print(f"  ascending={config.ascending}")
    print(f"  output_frame_limit={config.output_frame_limit}")
    print(f"  gifs={config.gifs}")

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
    histogram_bins = 80
    histogram_edges = np.linspace(0.0, TWO_PI, histogram_bins + 1)
    sweep_beta: Optional[float] = None
    sweep_expected_params: dict[float, str] = {}

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
            sweep_beta = beta
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
        mlp_null_title = r"\mathrm{MLP}\,=\,0"
        attention_label = "USA" if config.attention_mode == "unnormalized" else "SA"
        mlp_color = MLP_COLOR if config.attention_mode == "unnormalized" else SA_COLOR

        theta0_list = []
        for seed in seeds.particle_seeds:
            rng_particles = np.random.default_rng(seed)
            theta0_list.append(sample_theta0(rng_particles, config.n_particles))

        threshold = cluster_threshold(beta, config.cluster_scale)

        null_histories = []
        null_times = []
        null_steps = []
        null_stop_reasons = []
        null_cluster_times: list[Optional[float]] = []
        null_did_not_converge = False
        for idx, theta0 in enumerate(theta0_list):
            label = f"beta={beta} MLP_null init {idx + 1}/{config.num_point_inits}"
            bar = ProgressHandle(num_steps, label=label)
            if is_infinite:
                times, theta_hist, step_count, converged = _simulate_until_convergence(
                    theta0,
                    sim_config,
                    None,
                    threshold,
                    config.convergence_window,
                    config.max_steps,
                    config.convergence_drift_tol,
                    config.convergence_spread_factor,
                    progress=bar.update_to,
                    progress_every=progress_every,
                )
                if not converged:
                    null_did_not_converge = True
                stop_reason = "convergence" if converged else "max_steps"
            else:
                times, theta_hist = simulate(
                    theta0,
                    sim_config,
                    None,
                    progress=bar.update_to,
                    progress_every=progress_every,
                )
                step_count = num_steps
                stop_reason = "fixed_time"
            bar.close()
            null_histories.append(theta_hist)
            null_times.append(times)
            null_steps.append(step_count)
            null_stop_reasons.append(stop_reason)
            null_cluster_times.append(
                step_count * config.dt if stop_reason == "convergence" else None
            )
            print(f"  [null init {idx + 1}] stop_reason={stop_reason}")
            att0 = attention_drift_particles(
                theta_hist[0],
                beta,
                config.attention_mode,
                self_attention=config.self_attention,
                ascending=config.ascending,
            )
            att_end = attention_drift_particles(
                theta_hist[-1],
                beta,
                config.attention_mode,
                self_attention=config.self_attention,
                ascending=config.ascending,
            )
            max_att0 = float(np.max(np.abs(att0))) if att0.size else 0.0
            max_att_end = float(np.max(np.abs(att_end))) if att_end.size else 0.0
            print(
                f"  [null init {idx + 1}] t=0 max|att|={max_att0:.6e} "
                f"max|total|={max_att0:.6e}"
            )
            print(
                f"  [null init {idx + 1}] t=end max|att|={max_att_end:.6e} "
                f"max|total|={max_att_end:.6e}"
            )

        null_histogram_densities = [
            _histogram_density(hist[-1], histogram_edges) for hist in null_histories
        ]

        for mlp_scale_eff, params_json in scale_runs:
            run_dir = make_run_dir(experiment_dir, beta, params_json)
            print(f"Run directory: {run_dir}")
            run_start = time.perf_counter()
            did_not_converge = null_did_not_converge
            all_steps: list[int] = list(null_steps)

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
            )

            k_candidates = np.arange(0, config.k_max + 1)
            gamma_vals = gamma_k_s1(beta, k_candidates)
            k_max = int(k_candidates[np.argmax(gamma_vals)]) if k_candidates.size else 0
            fig = make_gamma_figure(beta, config.k_max, k_max)
            save_figure(fig, run_dir / "gamma_k", formats=("pdf",))
            import matplotlib.pyplot as plt

            plt.close(fig)

            mlp_params_list = []
            for mlp_seed in seeds.mlp_seeds:
                rng_mlp = np.random.default_rng(mlp_seed)
                mlp_params_list.append(sample_mlp_params(rng_mlp, mlp_config))

            print("Writing outputs...")
            if config.num_point_inits > 1:
                beta_label = f"{beta:.6g}"
                null_final = [hist[-1] for hist in null_histories]
                fig = make_total_clusters_figure(
                    null_final,
                    a=np.empty((0, 2)),
                    omega=np.empty((0, 2)),
                    activation=config.activation,
                    color=NULL_COLOR,
                    shade_regions=True,
                    show_potential=False,
                )
                save_figure(fig, run_dir / f"total_clusters_beta={beta_label}_null", formats=("pdf",))
                import matplotlib.pyplot as plt

                plt.close(fig)

            null_counts = []
            null_mode_counts = []
            null_mass_counts = []
            for j, theta_hist in enumerate(null_histories):
                label_slug = f"MLP_null_init{j + 1}"
                _save_frames_and_gif(
                    run_dir,
                    label_slug,
                    theta_hist,
                    null_times[j],
                    beta,
                    config.n_particles,
                    mlp_null_title,
                    attention_label,
                    a=np.empty((0, 2)),
                    omega=np.empty((0, 2)),
                    activation=config.activation,
                    color=NULL_COLOR,
                    plot_interval=config.plot_interval,
                    output_frame_limit=config.output_frame_limit,
                    show_potential=config.gradient_mlp,
                    save_gif=config.gifs,
                )
                null_suffix = "" if config.num_point_inits == 1 else f"_init{j + 1}"
                traj_null_stem = run_dir / f"trajectories_null{null_suffix}"
                fig = make_trajectory_figure(
                    null_times[j],
                    theta_hist,
                    color=NULL_COLOR,
                )
                save_figure(fig, traj_null_stem, formats=("pdf",))
                import matplotlib.pyplot as plt

                plt.close(fig)
                fig = make_trajectory_figure(
                    null_times[j],
                    theta_hist,
                    color=NULL_COLOR,
                    time_scale="log",
                )
                save_figure(fig, traj_null_stem.with_name(f"{traj_null_stem.name}_log"), formats=("pdf",))
                plt.close(fig)
                if is_infinite:
                    conv_idx = len(theta_hist) - 1
                else:
                    conv_idx = convergence_index(theta_hist, threshold, config.convergence_window)
                null_counts.append(cluster_count(theta_hist[conv_idx], threshold))
                null_mode_counts.append(mode_count(theta_hist[conv_idx], threshold))
                null_mass_counts.append(
                    mass_count(theta_hist[conv_idx], threshold, config.mass_threshold)
                )

            mlp_counts = []
            mlp_mode_counts = []
            mlp_mass_counts = []
            mlp_stop_reasons = []
            mlp_cluster_times: list[Optional[float]] = []
            mlp_histogram_densities: list[list[float]] = []
            for i, mlp_params in enumerate(mlp_params_list):
                mlp_histories = []
                mlp_times = []
                mlp_steps = []
                for j, theta0 in enumerate(theta0_list):
                    label = f"beta={beta} MLP{i + 1} init {j + 1}/{config.num_point_inits}"
                    bar = ProgressHandle(num_steps, label=label)
                    if is_infinite:
                        times, theta_hist, step_count, converged = _simulate_until_convergence(
                            theta0,
                            sim_config,
                            mlp_params,
                            threshold,
                            config.convergence_window,
                            config.max_steps,
                            config.convergence_drift_tol,
                            config.convergence_spread_factor,
                            progress=bar.update_to,
                            progress_every=progress_every,
                        )
                        if not converged:
                            did_not_converge = True
                        stop_reason = "convergence" if converged else "max_steps"
                    else:
                        times, theta_hist = simulate(
                            theta0,
                            sim_config,
                            mlp_params,
                            progress=bar.update_to,
                            progress_every=progress_every,
                        )
                        step_count = num_steps
                        stop_reason = "fixed_time"
                    bar.close()
                    mlp_histories.append(theta_hist)
                    mlp_times.append(times)
                    mlp_steps.append(step_count)
                    all_steps.append(step_count)
                    mlp_stop_reasons.append(stop_reason)
                    mlp_cluster_times.append(
                        step_count * config.dt if stop_reason == "convergence" else None
                    )
                    print(f"  [MLP {i + 1} init {j + 1}] stop_reason={stop_reason}")
                    att0 = attention_drift_particles(
                        theta_hist[0],
                        beta,
                        config.attention_mode,
                        self_attention=config.self_attention,
                        ascending=config.ascending,
                    )
                    mlp0 = mlp_drift(theta_hist[0], mlp_params)
                    total0 = att0 + mlp0
                    att_end = attention_drift_particles(
                        theta_hist[-1],
                        beta,
                        config.attention_mode,
                        self_attention=config.self_attention,
                        ascending=config.ascending,
                    )
                    mlp_end = mlp_drift(theta_hist[-1], mlp_params)
                    total_end = att_end + mlp_end
                    max_att0 = float(np.max(np.abs(att0))) if att0.size else 0.0
                    max_mlp0 = float(np.max(np.abs(mlp0))) if mlp0.size else 0.0
                    max_total0 = float(np.max(np.abs(total0))) if total0.size else 0.0
                    max_att_end = float(np.max(np.abs(att_end))) if att_end.size else 0.0
                    max_mlp_end = float(np.max(np.abs(mlp_end))) if mlp_end.size else 0.0
                    max_total_end = float(np.max(np.abs(total_end))) if total_end.size else 0.0
                    print(
                        f"  [MLP {i + 1} init {j + 1}] t=0 max|att|={max_att0:.6e} "
                        f"max|mlp|={max_mlp0:.6e} max|total|={max_total0:.6e}"
                    )
                    print(
                        f"  [MLP {i + 1} init {j + 1}] t=end max|att|={max_att_end:.6e} "
                        f"max|mlp|={max_mlp_end:.6e} max|total|={max_total_end:.6e}"
                    )

                for j, theta_hist in enumerate(mlp_histories):
                    label_parts = []
                    if config.num_mlp_inits > 1:
                        label_parts.append(f"MLP{i + 1}")
                    if config.num_point_inits > 1:
                        label_parts.append(f"init{j + 1}")
                    label_slug = "_".join(label_parts) if label_parts else "MLP"
                    _save_frames_and_gif(
                        run_dir,
                        label_slug,
                        theta_hist,
                        mlp_times[j],
                        beta,
                        config.n_particles,
                        mlp_title,
                        attention_label,
                        a=mlp_params.a,
                        omega=mlp_params.omega,
                        activation=mlp_params.activation,
                        color=mlp_color,
                        plot_interval=config.plot_interval,
                        output_frame_limit=config.output_frame_limit,
                        show_potential=config.gradient_mlp,
                        save_gif=config.gifs,
                    )
                    mlp_suffix = ""
                    if config.num_mlp_inits > 1:
                        mlp_suffix = f"_MLP{i + 1}"
                    if config.num_point_inits > 1:
                        mlp_suffix = f"{mlp_suffix}_init{j + 1}"
                    traj_mlp_stem = run_dir / f"trajectories_MLP{mlp_suffix}"
                    fig = make_trajectory_figure(
                        mlp_times[j],
                        theta_hist,
                        color=MLP_COLOR,
                    )
                    save_figure(fig, traj_mlp_stem, formats=("pdf",))
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                    fig = make_trajectory_figure(
                        mlp_times[j],
                        theta_hist,
                        color=MLP_COLOR,
                        time_scale="log",
                    )
                    save_figure(fig, traj_mlp_stem.with_name(f"{traj_mlp_stem.name}_log"), formats=("pdf",))
                    plt.close(fig)
                    hist_stem = run_dir / f"histogram{mlp_suffix}"
                    fig = make_histogram_figure(
                        theta_hist[-1],
                        null_histories[j][-1],
                        a=mlp_params.a,
                        omega=mlp_params.omega,
                        activation=mlp_params.activation,
                        include_null=False,
                        show_potential=config.gradient_mlp,
                        mlp_color=mlp_color,
                    )
                    save_figure(fig, hist_stem, formats=("pdf",))
                    plt.close(fig)
                    hist_null_stem = run_dir / f"histogram_with_null{mlp_suffix}"
                    fig = make_histogram_figure(
                        theta_hist[-1],
                        null_histories[j][-1],
                        a=mlp_params.a,
                        omega=mlp_params.omega,
                        activation=mlp_params.activation,
                        include_null=True,
                        show_potential=config.gradient_mlp,
                        mlp_color=mlp_color,
                    )
                    save_figure(fig, hist_null_stem, formats=("pdf",))
                    plt.close(fig)
                    if config.num_mlp_inits == 1 and config.num_point_inits == 1:
                        comparison_path = run_dir / "evolution_MLP_comparison.gif"
                    else:
                        comparison_path = run_dir / f"evolution_MLP_comparison_{label_slug}.gif"
                    if config.gifs:
                        _save_comparison_gif(
                            comparison_path,
                            null_histories[j],
                            null_times[j],
                            theta_hist,
                            mlp_times[j],
                            beta,
                            config.n_particles,
                            mlp_std_label,
                            attention_label,
                            a=mlp_params.a,
                            omega=mlp_params.omega,
                            activation=mlp_params.activation,
                            plot_interval=config.plot_interval,
                            output_frame_limit=config.output_frame_limit,
                            show_potential=config.gradient_mlp,
                            mlp_color=mlp_color,
                        )
                        _save_field_gif(
                            run_dir,
                            label_slug,
                            theta_hist,
                            mlp_times[j],
                            beta,
                            config.n_particles,
                            mlp_title,
                            attention_label,
                            config.attention_mode,
                            config.ascending,
                            a=mlp_params.a,
                            omega=mlp_params.omega,
                            activation=mlp_params.activation,
                            plot_interval=config.plot_interval,
                            output_frame_limit=config.output_frame_limit,
                            show_potential=config.gradient_mlp,
                        )
                    if is_infinite:
                        conv_idx = len(theta_hist) - 1
                    else:
                        conv_idx = convergence_index(theta_hist, threshold, config.convergence_window)
                    mlp_counts.append(cluster_count(theta_hist[conv_idx], threshold))
                    mlp_mode_counts.append(mode_count(theta_hist[conv_idx], threshold))
                    mlp_mass_counts.append(
                        mass_count(theta_hist[conv_idx], threshold, config.mass_threshold)
                    )
                    mlp_histogram_densities.append(
                        _histogram_density(theta_hist[-1], histogram_edges)
                    )

                if config.num_point_inits > 1:
                    beta_label = f"{beta:.6g}"
                    suffix = f"_MLP{i + 1}" if config.num_mlp_inits > 1 else ""
                    mlp_final = [hist[-1] for hist in mlp_histories]
                    fig = make_total_clusters_figure(
                        mlp_final,
                        a=mlp_params.a,
                        omega=mlp_params.omega,
                        activation=mlp_params.activation,
                        color=mlp_color,
                        shade_regions=False,
                        show_potential=config.gradient_mlp,
                    )
                    save_figure(fig, run_dir / f"total_clusters_beta={beta_label}{suffix}", formats=("pdf",))
                    import matplotlib.pyplot as plt

                    plt.close(fig)

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
                null_counts,
                mlp_counts,
                null_mode_counts,
                mlp_mode_counts,
                null_mass_counts,
                mlp_mass_counts,
                null_stop_reasons,
                mlp_stop_reasons,
                config.num_mlp_inits,
                config.num_point_inits,
                run_seconds,
                mlp_scale_eff,
                config.mlp_scale_mode,
                null_cluster_times,
                mlp_cluster_times,
                histogram_edges.tolist(),
                null_histogram_densities,
                mlp_histogram_densities,
            )

            if did_not_converge:
                print("Warning: convergence not reached before max_steps.")

            print(f"Saved results to {run_dir}")

    if multi_scale and sweep_expected_params:
        entries = _load_scale_summaries(experiment_dir, sweep_expected_params)
        if entries:
            payload_entries = []
            for scale, data, run_dir in entries:
                entry = dict(data)
                entry["run_dir"] = str(run_dir)
                payload_entries.append(entry)
            payload = {
                "beta": sweep_beta,
                "mlp_scale_mode": config.mlp_scale_mode,
                "entries": payload_entries,
            }
            write_json(experiment_dir / "mlp_scale_sweep.json", payload, compact=False)
            by_beta: dict[float, list[tuple[float, dict, Path]]] = {}
            for scale, data, run_dir in entries:
                try:
                    beta_value = float(data.get("beta", sweep_beta))
                except (TypeError, ValueError):
                    continue
                by_beta.setdefault(beta_value, []).append((scale, data, run_dir))

            for beta_value, items in by_beta.items():
                items.sort(key=lambda item: item[0])
                scales = np.asarray([item[0] for item in items], dtype=float)
                stop_times = np.asarray(
                    [_stop_time(item[1].get("mlp_cluster_times", [])) for item in items],
                    dtype=float,
                )
                fig = make_mlp_scale_stop_time_figure(scales, stop_times)
                output_dir = items[0][2]
                save_figure(fig, output_dir / "mlp_scale_stop_time", formats=("pdf",))
                import matplotlib.pyplot as plt

                plt.close(fig)

    if multi_scale:
        return

    summaries = _load_run_summaries(experiment_dir, expected_params)
    if summaries:
        sqrt_betas = np.asarray([float(np.sqrt(float(entry["beta"]))) for entry in summaries])
        order = np.argsort(sqrt_betas)
        sqrt_betas = sqrt_betas[order]
        null_counts = [entry.get("null_counts", []) for entry in summaries]
        mlp_counts = [entry.get("mlp_counts", []) for entry in summaries]
        null_mode_counts = [entry.get("null_mode_counts", []) for entry in summaries]
        mlp_mode_counts = [entry.get("mlp_mode_counts", []) for entry in summaries]
        null_mass_counts = [entry.get("null_mass_counts", []) for entry in summaries]
        mlp_mass_counts = [entry.get("mlp_mass_counts", []) for entry in summaries]

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

        null_mean, _ = _stats(null_mode_counts)
        mlp_mean, _ = _stats(mlp_mode_counts)
        fig = make_cluster_bar_plot(
            sqrt_betas,
            mlp_mean,
            ylabel="mode count",
        )
        save_figure(fig, stats_dir / "mode_count", formats=("pdf",))
        plt.close(fig)
        fig = make_cluster_bar_plot_with_null(
            sqrt_betas,
            null_mean,
            mlp_mean,
            ylabel="mode count",
        )
        save_figure(fig, stats_dir / "mode_count_with_null", formats=("pdf",))
        plt.close(fig)

        null_mean, _ = _stats(null_mass_counts)
        mlp_mean, _ = _stats(mlp_mass_counts)
        fig = make_cluster_bar_plot(
            sqrt_betas,
            mlp_mean,
            ylabel="mass count",
        )
        save_figure(fig, stats_dir / "mass_count", formats=("pdf",))
        plt.close(fig)
        fig = make_cluster_bar_plot_with_null(
            sqrt_betas,
            null_mean,
            mlp_mean,
            ylabel="mass count",
        )
        save_figure(fig, stats_dir / "mass_count_with_null", formats=("pdf",))
        plt.close(fig)


def main(config_path: Optional[Path] = None) -> None:
    path = config_path or Path("config.json")
    config = load_config(path)
    run_experiment(config)


if __name__ == "__main__":
    main()
