"""Particle dynamics on S2 with self-attention and optional MLP drift."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

Activation = Literal["relu", "gelu"]
ScaleMode = Literal["std", "norm"]
AttentionMode = Literal["unnormalized", "normalized"]
Integrator = Literal["euler", "rk2", "rk4"]


@dataclass(frozen=True)
class MLPParams:
    a: NDArray[np.float64]
    omega: NDArray[np.float64]
    activation: Activation


@dataclass(frozen=True)
class SimulationConfig:
    beta: float
    dt: float
    num_steps: int
    save_every: int
    attention_mode: AttentionMode
    self_attention: bool
    ascending: bool
    integrator: Integrator


@dataclass(frozen=True)
class MLPConfig:
    n_units: int
    activation: Activation
    weight_scale: float
    weight_scale_mode: ScaleMode
    gradient_mlp: bool
    dimension: int


def _activation(z: NDArray[np.float64], activation: Activation) -> NDArray[np.float64]:
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    raise ValueError(f"Unsupported activation: {activation}")


def _normalize_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return values / norms


def _normalize_rows_safe(
    values: NDArray[np.float64],
    fallback: NDArray[np.float64],
) -> NDArray[np.float64]:
    if values.size == 0:
        return values
    norms = np.linalg.norm(values, axis=1)
    finite = np.isfinite(values).all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
    if np.all(finite):
        return values / norms[:, None]
    safe_norms = np.where(finite, norms, 1.0)
    normalized = values / safe_norms[:, None]
    normalized[~finite] = fallback[~finite]
    return normalized


def sample_points_on_sphere(
    rng: np.random.Generator, n_particles: int, dimension: int
) -> NDArray[np.float64]:
    """Sample uniform points on S^{dimension-1}."""
    if dimension < 2:
        raise ValueError("dimension must be >= 2 for sphere sampling.")
    points = rng.normal(size=(n_particles, dimension))
    return _normalize_rows(points)


def sample_mlp_params(rng: np.random.Generator, config: MLPConfig) -> MLPParams:
    """Sample MLP parameters for S2 dynamics.

    The drift is u(x) = proj_x sum_j omega_j * sigma(a_j dot x).
    """
    a = rng.normal(size=(config.n_units, config.dimension))
    a /= np.linalg.norm(a, axis=1, keepdims=True)

    if config.weight_scale_mode == "std":
        if config.gradient_mlp:
            scalars = rng.normal(scale=config.weight_scale, size=(config.n_units, 1))
            omega = scalars * a
        else:
            omega = rng.normal(scale=config.weight_scale, size=(config.n_units, config.dimension))
    elif config.weight_scale_mode == "norm":
        if config.weight_scale <= 0.0:
            omega = np.zeros((config.n_units, config.dimension))
        elif config.gradient_mlp:
            signs = rng.normal(size=(config.n_units, 1))
            signs = np.where(signs >= 0.0, 1.0, -1.0)
            omega = config.weight_scale * signs * a
        else:
            omega = rng.normal(size=(config.n_units, config.dimension))
            norms = np.linalg.norm(omega, axis=1, keepdims=True)
            norms = np.where(norms > 0.0, norms, 1.0)
            omega = config.weight_scale * omega / norms
    else:
        raise ValueError(f"Unsupported weight_scale_mode: {config.weight_scale_mode}")
    return MLPParams(a=a.astype(np.float64), omega=omega.astype(np.float64), activation=config.activation)


def _attention_drift_vectors(
    x_eval: NDArray[np.float64],
    x_particles: NDArray[np.float64],
    beta: float,
    mode: AttentionMode,
    self_attention: bool,
    ascending: bool,
) -> NDArray[np.float64]:
    dots = x_eval @ x_particles.T
    log_w = beta * (dots - 1.0)
    if (not self_attention) and (x_eval.shape[0] == x_particles.shape[0]):
        np.fill_diagonal(log_w, -np.inf)

    weights = np.exp(np.clip(log_w, -80.0, 0.0))
    if (not self_attention) and (x_eval.shape[0] == x_particles.shape[0]):
        np.fill_diagonal(weights, 0.0)

    weighted_sum = weights @ x_particles
    dot_sum = (weights * dots).sum(axis=1, keepdims=True)
    numerator = dot_sum * x_eval - weighted_sum
    sign = -1.0 if ascending else 1.0

    if mode == "normalized":
        denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        return sign * numerator / denom

    return sign * numerator / max(1, x_particles.shape[0])


def attention_drift_particles_vectors(
    x: NDArray[np.float64],
    beta: float,
    mode: AttentionMode,
    self_attention: bool = False,
    ascending: bool = False,
) -> NDArray[np.float64]:
    """Compute self-attention drift on S^{d-1} for vector positions."""
    effective_self_attention = self_attention or mode == "normalized"
    return _attention_drift_vectors(
        x,
        x,
        beta,
        mode,
        effective_self_attention,
        ascending,
    )


def mlp_drift_vectors(x: NDArray[np.float64], params: MLPParams) -> NDArray[np.float64]:
    """Compute the MLP drift contribution as a tangent vector field."""
    z = x @ params.a.T
    act = _activation(z, params.activation)
    v = act @ params.omega
    dot = np.einsum("ij,ij->i", v, x)
    return v - dot[:, None] * x


def _total_drift_vectors(
    x: NDArray[np.float64],
    sim_config: SimulationConfig,
    mlp_params: Optional[MLPParams],
) -> NDArray[np.float64]:
    drift = attention_drift_particles_vectors(
        x,
        sim_config.beta,
        sim_config.attention_mode,
        self_attention=sim_config.self_attention,
        ascending=sim_config.ascending,
    )
    if mlp_params is not None:
        drift += mlp_drift_vectors(x, mlp_params)
    return drift


def step_positions(
    x: NDArray[np.float64],
    sim_config: SimulationConfig,
    mlp_params: Optional[MLPParams],
) -> NDArray[np.float64]:
    """Advance one time step for S^{d-1} positions."""
    dt = sim_config.dt
    x0 = _normalize_rows(x)
    if sim_config.integrator == "euler":
        drift = _total_drift_vectors(x0, sim_config, mlp_params)
        x_next = x0 + dt * drift
    elif sim_config.integrator == "rk2":
        k1 = _total_drift_vectors(x0, sim_config, mlp_params)
        x_mid = _normalize_rows(x0 + 0.5 * dt * k1)
        k2 = _total_drift_vectors(x_mid, sim_config, mlp_params)
        x_next = x0 + dt * k2
    elif sim_config.integrator == "rk4":
        k1 = _total_drift_vectors(x0, sim_config, mlp_params)
        x_mid1 = _normalize_rows(x0 + 0.5 * dt * k1)
        k2 = _total_drift_vectors(x_mid1, sim_config, mlp_params)
        x_mid2 = _normalize_rows(x0 + 0.5 * dt * k2)
        k3 = _total_drift_vectors(x_mid2, sim_config, mlp_params)
        x_end = _normalize_rows(x0 + dt * k3)
        k4 = _total_drift_vectors(x_end, sim_config, mlp_params)
        x_next = x0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        raise ValueError(f"Unknown integrator: {sim_config.integrator}")
    return _normalize_rows_safe(x_next, x0)


def simulate_positions(
    x0: NDArray[np.float64],
    sim_config: SimulationConfig,
    mlp_params: Optional[MLPParams],
    progress: Optional[Callable[[int, int], None]] = None,
    progress_every: Optional[int] = None,
    save_history: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run integration and return (times, position_history).
    
    If save_history=False, only returns initial and final states to save memory.
    """
    x = _normalize_rows(x0.astype(np.float64, copy=True))

    if save_history:
        times = [0.0]
        history = [x.copy()]
    else:
        # Only keep initial state, will add final at the end
        times = [0.0]
        history = [x.copy()]

    if progress is not None:
        if progress_every is None:
            progress_every = max(1, sim_config.num_steps // 100)
        progress(0, sim_config.num_steps)

    for step in range(1, sim_config.num_steps + 1):
        x = step_positions(x, sim_config, mlp_params)

        if save_history and (step % sim_config.save_every == 0 or step == sim_config.num_steps):
            times.append(step * sim_config.dt)
            history.append(x.copy())

        if progress is not None and (step % progress_every == 0 or step == sim_config.num_steps):
            progress(step, sim_config.num_steps)

    if not save_history:
        # Only add final state
        times.append(sim_config.num_steps * sim_config.dt)
        history.append(x.copy())

    return np.asarray(times), np.asarray(history)
