"""Configuration loading and seed planning for S2 simulations."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from wgf_circle.mlp_params_io import load_mlp_params_file

Activation = Literal["relu", "gelu"]

@dataclass(frozen=True)
class RunConfig:
    betas: List[float]
    n_particles: int
    dt: float
    total_time: float
    save_every: int
    k_max: int
    num_mlp_inits: int
    num_point_inits: int
    mlp_units: int
    mlp_units_from_dimension: bool
    activation: Activation
    mlp_scale: float
    mlp_scales: List[float]
    mlp_scale_mode: str
    gradient_mlp: bool
    particle_seed: int
    mlp_seed: int
    mlp_params_path: Optional[Path]
    mlp_params: Optional[List[Tuple[np.ndarray, np.ndarray, str]]]
    mlp_params_hash: Optional[str]
    results_dir: Path
    dimension: int
    dimensions: List[int]
    plot_interval: float
    cluster_scale: float
    mass_threshold: float
    convergence_window: int
    attention_mode: str
    integrator: str
    max_steps: int
    self_attention: bool
    ascending: bool
    convergence_drift_tol: float
    convergence_spread_factor: float
    output_frame_limit: int
    mlp0_output_frame_limit: Optional[int]
    gif_sphere: bool
    gif_histogram: bool
    pdf_trajectory: bool
    experiment_dir: Optional[Path]
    sphere_html_view: str
    sphere_gif_rotations: float


@dataclass(frozen=True)
class SeedPlan:
    particle_seeds: List[int]
    mlp_seeds: List[int]


DEFAULT_CONFIG: Dict[str, Any] = {
    "betas": [0.1, 0.5, 1, 2, 5, 7, 9, 12, 15, 25],
    "n_particles": 1000,
    "dt": 5e-4,
    "total_time": 20.0,
    "save_every": 10,
    "k_max": 20,
    "num_mlp_inits": 1,
    "num_point_inits": 1,
    "mlp_units": None,
    "activation": "relu",
    "mlp_scale": 0.5,
    "mlp_scale_mode": "std",
    "gradient_MLP": True,
    "mlp_params_path": None,
    "dimension": 3,
    "attention_mode": "unnormalized",
    "integrator": "euler",
    "particle_seed": 7,
    "mlp_seed": 11,
    "results_dir": "results",
    "plot_interval": 0.1,
    "cluster_scale": 1.0,
    "mass_threshold": 0.0,
    "convergence_window": 5,
    "max_steps": 200000,
    "self_attention": False,
    "ascending": False,
    "convergence_drift_tol": 1e-3,
    "convergence_spread_factor": 1.0,
    "output_frame_limit": 400,
    "mlp0_output_frame_limit": None,
    "gif_sphere": True,
    "gif_histogram": True,
    "pdf_trajectory": True,
    "experiment_dir": None,
    "sphere_html_view": "mlp",
    "sphere_gif_rotations": 1.0,
}


def _parse_betas(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        return [float(b.strip()) for b in value.split(",") if b.strip()]
    raise ValueError("betas must be a list or comma-separated string.")


def _parse_mlp_scales(value: Any) -> List[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        scales = [float(v) for v in value]
        if len(scales) == 3:
            start, stop, step = scales
            if step == 0.0:
                raise ValueError("mlp_scale range step must be non-zero.")
            if start == stop:
                return [start]
            step = abs(step) * (1.0 if stop > start else -1.0)
            seq = np.arange(start, stop + 0.5 * step, step)
            if step > 0:
                seq = seq[seq <= stop + 1e-12]
            else:
                seq = seq[seq >= stop - 1e-12]
            return [float(x) for x in seq]
    else:
        scales = [float(value)]
    if not scales:
        raise ValueError("mlp_scale must be a number or a non-empty list.")
    return scales


def _parse_dimensions(value: Any) -> List[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        dims = [int(v) for v in value]
    else:
        dims = [int(value)]
    if not dims:
        raise ValueError("dimension must be an integer or a non-empty list.")
    ordered: List[int] = []
    seen = set()
    for dim in dims:
        if dim not in seen:
            ordered.append(dim)
            seen.add(dim)
    return ordered


def _parse_total_time(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"inf", "infty", "infinite", "infinity"}:
            return float("inf")
    return float(value)


def load_config(path: Path) -> RunConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")

    merged: Dict[str, Any] = dict(DEFAULT_CONFIG)
    merged.update(data)
    if "exclude_self" in data:
        raise ValueError("exclude_self was renamed to self_attention.")
    if "tie_potential" in data:
        raise ValueError("tie_potential was renamed to gradient_MLP.")
    gradient_mlp = merged.get("gradient_MLP", True)
    if not isinstance(gradient_mlp, bool):
        raise ValueError("gradient_MLP must be a boolean.")
    merged["gradient_MLP"] = gradient_mlp

    betas = _parse_betas(merged["betas"])
    total_time = _parse_total_time(merged["total_time"])
    if not betas:
        raise ValueError("At least one beta value is required.")
    dimensions = _parse_dimensions(merged["dimension"])
    if any(dim < 2 for dim in dimensions):
        raise ValueError("dimension must be >= 2.")
    merged["dimension"] = int(dimensions[0])
    if merged["n_particles"] <= 0:
        raise ValueError("n_particles must be positive.")
    if merged["dt"] <= 0.0:
        raise ValueError("dt must be positive.")
    if total_time <= 0.0 and not np.isinf(total_time):
        raise ValueError("total_time must be positive or inf.")
    if merged["save_every"] <= 0:
        raise ValueError("save_every must be positive.")
    if merged["num_mlp_inits"] <= 0:
        raise ValueError("num_mlp_inits must be positive.")
    if merged["num_point_inits"] <= 0:
        raise ValueError("num_point_inits must be positive.")
    if merged["plot_interval"] <= 0.0:
        raise ValueError("plot_interval must be positive.")
    if merged["cluster_scale"] <= 0.0:
        raise ValueError("cluster_scale must be positive.")
    if merged["convergence_window"] <= 0:
        raise ValueError("convergence_window must be positive.")
    attention_mode = str(merged.get("attention_mode", "unnormalized")).strip().lower()
    if attention_mode not in {"unnormalized", "normalized"}:
        raise ValueError("attention_mode must be 'unnormalized' or 'normalized'.")
    activation = str(merged.get("activation", "relu")).strip().lower()
    if activation not in {"relu", "gelu"}:
        raise ValueError("activation must be 'relu' or 'gelu'.")
    mlp_scale_mode = str(merged.get("mlp_scale_mode", "std")).strip().lower()
    if mlp_scale_mode not in {"std", "norm"}:
        raise ValueError("mlp_scale_mode must be 'std' or 'norm'.")
    integrator = str(merged.get("integrator", "euler")).strip().lower()
    if integrator not in {"euler", "rk2", "rk4"}:
        raise ValueError("integrator must be one of: euler, rk2, rk4.")
    if merged["max_steps"] <= 0:
        raise ValueError("max_steps must be positive.")
    if merged["output_frame_limit"] <= 0:
        raise ValueError("output_frame_limit must be positive.")
    if merged["convergence_spread_factor"] < 0.0:
        raise ValueError("convergence_spread_factor must be non-negative.")
    if merged["convergence_drift_tol"] < 0.0:
        raise ValueError("convergence_drift_tol must be non-negative.")
    mlp0_output_frame_limit = merged.get("mlp0_output_frame_limit")
    if mlp0_output_frame_limit in (None, ""):
        mlp0_output_frame_limit = None
    else:
        mlp0_output_frame_limit = int(mlp0_output_frame_limit)
        if mlp0_output_frame_limit <= 0:
            raise ValueError("mlp0_output_frame_limit must be positive or null.")
    merged["mlp0_output_frame_limit"] = mlp0_output_frame_limit
    if not isinstance(merged["self_attention"], bool):
        raise ValueError("self_attention must be a boolean.")
    if not isinstance(merged["ascending"], bool):
        raise ValueError("ascending must be a boolean.")
    if "gif_sphere" not in data and "gifs" in data:
        merged["gif_sphere"] = bool(data["gifs"])
    if "gif_histogram" not in data and "gifs" in data:
        merged["gif_histogram"] = bool(data["gifs"])
    gif_sphere = merged.get("gif_sphere", DEFAULT_CONFIG["gif_sphere"])
    gif_histogram = merged.get("gif_histogram", DEFAULT_CONFIG["gif_histogram"])
    if not isinstance(gif_sphere, bool):
        raise ValueError("gif_sphere must be a boolean.")
    if not isinstance(gif_histogram, bool):
        raise ValueError("gif_histogram must be a boolean.")
    merged["gif_sphere"] = gif_sphere
    merged["gif_histogram"] = gif_histogram
    if len(dimensions) == 1 and merged["dimension"] != 3:
        warnings.warn(
            "dimension != 3: disabling sphere plots/gifs/trajectories; stats-only run.",
            RuntimeWarning,
        )
        merged["gif_sphere"] = False
        merged["gif_histogram"] = False
        merged["pdf_trajectory"] = False
    sphere_html_view = str(merged.get("sphere_html_view", "mlp")).strip().lower()
    if sphere_html_view not in {"mlp", "null"}:
        raise ValueError("sphere_html_view must be 'mlp' or 'null'.")
    merged["sphere_html_view"] = sphere_html_view
    sphere_gif_rotations = float(merged.get("sphere_gif_rotations", 1.0))
    if sphere_gif_rotations <= 0.0:
        raise ValueError("sphere_gif_rotations must be positive.")
    merged["sphere_gif_rotations"] = sphere_gif_rotations

    mlp_params_path = merged.get("mlp_params_path")
    if mlp_params_path in (None, ""):
        mlp_params_path = None
    else:
        mlp_params_path = Path(mlp_params_path).expanduser()
        if not mlp_params_path.is_absolute():
            mlp_params_path = (path.parent / mlp_params_path).resolve()
        if not mlp_params_path.exists():
            raise FileNotFoundError(f"MLP params file not found: {mlp_params_path}")

    mlp_params = None
    mlp_params_hash = None
    if mlp_params_path is not None:
        if len(dimensions) > 1:
            raise ValueError("mlp_params_path cannot be used with multiple dimensions.")
        mlp_params, mlp_params_hash = load_mlp_params_file(
            mlp_params_path,
            merged["dimension"],
            activation,
        )
        activations = {entry[2] for entry in mlp_params}
        if len(activations) != 1:
            raise ValueError("mlp_params_path must use a single activation.")
        activation = activations.pop()
        if activation != str(merged.get("activation", "")).strip().lower():
            warnings.warn("activation overridden by mlp_params_path.", RuntimeWarning)
        unit_counts = {entry[0].shape[0] for entry in mlp_params}
        if len(unit_counts) != 1:
            raise ValueError("mlp_params_path must use a consistent number of units.")
        mlp_units = unit_counts.pop()
        if len(mlp_params) > 1:
            merged["num_mlp_inits"] = len(mlp_params)
        elif merged["num_mlp_inits"] != 1:
            raise ValueError(
                "mlp_params_path provides a single MLP; set num_mlp_inits=1 or provide a list."
            )
        mlp_units_from_dimension = False
    else:
        if merged["mlp_units"] is None:
            mlp_units = merged["dimension"]
            mlp_units_from_dimension = True
        else:
            mlp_units = int(merged["mlp_units"])
            mlp_units_from_dimension = False
    if mlp_params is not None:
        mlp_units_from_dimension = False
    mlp_scales = _parse_mlp_scales(merged["mlp_scale"])
    if any(scale < 0.0 for scale in mlp_scales):
        raise ValueError("mlp_scale entries must be non-negative.")
    if len(mlp_scales) > 1 and len(betas) != 1:
        raise ValueError("mlp_scale list requires a single beta value.")
    if mlp_params is not None and len(mlp_scales) > 1:
        raise ValueError("mlp_params_path cannot be combined with an mlp_scale list.")
    merged["mlp_scale"] = float(mlp_scales[0])
    if merged["mass_threshold"] < 0.0 or merged["mass_threshold"] > 1.0:
        raise ValueError("mass_threshold must be between 0 and 1.")
    if merged["particle_seed"] is None or merged["mlp_seed"] is None:
        raise ValueError("particle_seed and mlp_seed must be set.")

    if np.isinf(total_time):
        num_steps = int(merged["max_steps"])
    else:
        num_steps = int(round(total_time / merged["dt"]))
    step_warn = 200_000
    particle_warn = 5_000
    work_warn = 50_000_000
    work_estimate = merged["n_particles"] * max(1, num_steps)
    if (
        merged["n_particles"] > particle_warn
        or num_steps > step_warn
        or work_estimate > work_warn
    ):
        warnings.warn(
            "Large run detected (n_particles, num_steps, or their product is high); "
            "this may be slow or memory intensive.",
            RuntimeWarning,
        )

    experiment_dir = merged.get("experiment_dir")
    if experiment_dir in (None, ""):
        experiment_dir = None

    return RunConfig(
        betas=betas,
        n_particles=merged["n_particles"],
        dt=merged["dt"],
        total_time=total_time,
        save_every=merged["save_every"],
        k_max=merged["k_max"],
        num_mlp_inits=merged["num_mlp_inits"],
        num_point_inits=merged["num_point_inits"],
        mlp_units=mlp_units,
        mlp_units_from_dimension=mlp_units_from_dimension,
        activation=activation,
        mlp_scale=merged["mlp_scale"],
        mlp_scales=mlp_scales,
        mlp_scale_mode=mlp_scale_mode,
        gradient_mlp=gradient_mlp,
        particle_seed=int(merged["particle_seed"]),
        mlp_seed=int(merged["mlp_seed"]),
        mlp_params_path=mlp_params_path,
        mlp_params=mlp_params,
        mlp_params_hash=mlp_params_hash,
        results_dir=Path(merged["results_dir"]),
        dimension=merged["dimension"],
        dimensions=dimensions,
        plot_interval=merged["plot_interval"],
        cluster_scale=merged["cluster_scale"],
        mass_threshold=merged["mass_threshold"],
        convergence_window=merged["convergence_window"],
        attention_mode=attention_mode,
        integrator=integrator,
        max_steps=int(merged["max_steps"]),
        self_attention=bool(merged["self_attention"]),
        ascending=bool(merged["ascending"]),
        convergence_drift_tol=float(merged["convergence_drift_tol"]),
        convergence_spread_factor=float(merged["convergence_spread_factor"]),
        output_frame_limit=int(merged["output_frame_limit"]),
        mlp0_output_frame_limit=merged["mlp0_output_frame_limit"],
        gif_sphere=bool(merged["gif_sphere"]),
        gif_histogram=bool(merged["gif_histogram"]),
        pdf_trajectory=bool(merged.get("pdf_trajectory", DEFAULT_CONFIG["pdf_trajectory"])),
        experiment_dir=Path(experiment_dir) if experiment_dir is not None else None,
        sphere_html_view=sphere_html_view,
        sphere_gif_rotations=float(merged["sphere_gif_rotations"]),
    )


def build_seed_plan(config: RunConfig) -> Dict[float, SeedPlan]:
    plan: Dict[float, SeedPlan] = {}
    particle_rng = np.random.default_rng(config.particle_seed)
    mlp_rng = np.random.default_rng(config.mlp_seed)
    global_particle_seeds = [config.particle_seed]
    global_mlp_seeds = [config.mlp_seed]
    if config.num_point_inits > 1:
        extra = config.num_point_inits - 1
        global_particle_seeds.extend(
            particle_rng.integers(0, 2**31 - 1, size=extra, dtype=np.int64).tolist()
        )
    if config.num_mlp_inits > 1:
        extra = config.num_mlp_inits - 1
        global_mlp_seeds.extend(
            mlp_rng.integers(0, 2**31 - 1, size=extra, dtype=np.int64).tolist()
        )
    for beta in config.betas:
        plan[beta] = SeedPlan(
            particle_seeds=global_particle_seeds,
            mlp_seeds=global_mlp_seeds,
        )
    return plan
