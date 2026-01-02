"""Particle dynamics on S1 with self-attention and optional MLP drift."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

TWO_PI = 2.0 * np.pi

Activation = Literal["relu", "gelu"]
AttentionMode = Literal["unnormalized", "normalized"]
Integrator = Literal["euler", "rk2", "rk4"]
UnnormalizedScaleMode = Literal["standard", "minus_beta"]


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
    unnormalized_scale_mode: UnnormalizedScaleMode
    exclude_self: bool
    integrator: Integrator


@dataclass(frozen=True)
class MLPConfig:
    n_units: int
    activation: Activation
    weight_scale: float
    tie_potential: bool


def _activation(z: NDArray[np.float64], activation: Activation) -> NDArray[np.float64]:
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    raise ValueError(f"Unsupported activation: {activation}")


def sample_theta0(rng: np.random.Generator, n_particles: int) -> NDArray[np.float64]:
    """Sample uniform angles on S1."""
    return rng.uniform(0.0, TWO_PI, size=n_particles)


def sample_mlp_params(rng: np.random.Generator, config: MLPConfig) -> MLPParams:
    """Sample MLP parameters for S1 dynamics.

    The drift is u(x) = proj_x sum_j omega_j * sigma(a_j dot x).
    """
    if not config.tie_potential:
        raise ValueError("This simulator enforces a gradient MLP (tie_potential=True).")
    a = rng.normal(size=(config.n_units, 2))
    a /= np.linalg.norm(a, axis=1, keepdims=True)

    scalars = rng.normal(scale=config.weight_scale, size=(config.n_units, 1))
    omega = scalars * a
    return MLPParams(a=a.astype(np.float64), omega=omega.astype(np.float64), activation=config.activation)


def attention_drift(
    theta: NDArray[np.float64],
    beta: float,
    mode: AttentionMode,
    unnormalized_scale_mode: UnnormalizedScaleMode = "standard",
    exclude_self: bool = True,
) -> NDArray[np.float64]:
    """Compute the self-attention drift on S1 (USA/SA model in angle form)."""
    effective_exclude = exclude_self and mode != "normalized"
    diff = theta[:, None] - theta[None, :]
    arg = beta * np.cos(diff)
    if effective_exclude:
        np.fill_diagonal(arg, -np.inf)

    row_max = np.max(arg, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    scaled = np.exp(np.clip(arg - row_max, -80.0, 80.0))
    if effective_exclude:
        np.fill_diagonal(scaled, 0.0)

    weighted = (scaled * np.sin(diff)).sum(axis=1)
    if mode == "normalized":
        norm = scaled.sum(axis=1)
        norm = np.maximum(norm, 1e-12)
        drift = -weighted / norm
    else:
        scale_offset = -beta if unnormalized_scale_mode == "minus_beta" else 0.0
        scale = np.exp(np.clip(row_max.squeeze() + scale_offset, -80.0, 80.0))
        weighted = weighted * scale
        drift = -weighted / max(1, theta.size)
    return drift


def attention_drift_field(
    theta_eval: NDArray[np.float64],
    theta_particles: NDArray[np.float64],
    beta: float,
    mode: AttentionMode,
    unnormalized_scale_mode: UnnormalizedScaleMode = "standard",
) -> NDArray[np.float64]:
    """Evaluate attention drift at arbitrary angles against a particle set."""
    diff = theta_eval[:, None] - theta_particles[None, :]
    arg = beta * np.cos(diff)
    row_max = np.max(arg, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    scaled = np.exp(np.clip(arg - row_max, -80.0, 80.0))
    weighted = (scaled * np.sin(diff)).sum(axis=1)
    if mode == "normalized":
        norm = scaled.sum(axis=1)
        norm = np.maximum(norm, 1e-12)
        drift = -weighted / norm
    else:
        scale_offset = -beta if unnormalized_scale_mode == "minus_beta" else 0.0
        scale = np.exp(np.clip(row_max.squeeze() + scale_offset, -80.0, 80.0))
        weighted = weighted * scale
        drift = -weighted / max(1, theta_particles.size)
    return drift


def mlp_drift(theta: NDArray[np.float64], params: MLPParams) -> NDArray[np.float64]:
    """Compute the MLP drift contribution as an angular velocity."""
    x = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    t = np.stack([-np.sin(theta), np.cos(theta)], axis=1)

    z = x @ params.a.T
    act = _activation(z, params.activation)
    v = act @ params.omega
    return np.einsum("ij,ij->i", t, v)


def simulate(
    theta0: NDArray[np.float64],
    sim_config: SimulationConfig,
    mlp_params: Optional[MLPParams],
    progress: Optional[Callable[[int, int], None]] = None,
    progress_every: Optional[int] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run Euler integration and return (times, theta_history)."""
    theta = theta0.astype(np.float64, copy=True)

    times = [0.0]
    history = [theta.copy()]

    if progress is not None:
        if progress_every is None:
            progress_every = max(1, sim_config.num_steps // 100)
        progress(0, sim_config.num_steps)

    for step in range(1, sim_config.num_steps + 1):
        theta = step_theta(theta, sim_config, mlp_params)

        if step % sim_config.save_every == 0 or step == sim_config.num_steps:
            times.append(step * sim_config.dt)
            history.append(theta.copy())

        if progress is not None and (step % progress_every == 0 or step == sim_config.num_steps):
            progress(step, sim_config.num_steps)

    return np.asarray(times), np.asarray(history)


def _total_drift(
    theta: NDArray[np.float64],
    sim_config: SimulationConfig,
    mlp_params: Optional[MLPParams],
) -> NDArray[np.float64]:
    drift = attention_drift(
        theta,
        sim_config.beta,
        sim_config.attention_mode,
        sim_config.unnormalized_scale_mode,
        exclude_self=sim_config.exclude_self,
    )
    if mlp_params is not None:
        drift += mlp_drift(theta, mlp_params)
    return drift


def step_theta(
    theta: NDArray[np.float64],
    sim_config: SimulationConfig,
    mlp_params: Optional[MLPParams],
) -> NDArray[np.float64]:
    """Advance one time step using the configured integrator."""
    dt = sim_config.dt
    if sim_config.integrator == "euler":
        drift = _total_drift(theta, sim_config, mlp_params)
        theta_next = theta + dt * drift
    elif sim_config.integrator == "rk2":
        k1 = _total_drift(theta, sim_config, mlp_params)
        k2 = _total_drift(theta + 0.5 * dt * k1, sim_config, mlp_params)
        theta_next = theta + dt * k2
    elif sim_config.integrator == "rk4":
        k1 = _total_drift(theta, sim_config, mlp_params)
        k2 = _total_drift(theta + 0.5 * dt * k1, sim_config, mlp_params)
        k3 = _total_drift(theta + 0.5 * dt * k2, sim_config, mlp_params)
        k4 = _total_drift(theta + dt * k3, sim_config, mlp_params)
        theta_next = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        raise ValueError(f"Unknown integrator: {sim_config.integrator}")
    return np.mod(theta_next, TWO_PI)


