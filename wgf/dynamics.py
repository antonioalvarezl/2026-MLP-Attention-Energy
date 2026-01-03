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
    gradient_mlp: bool


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
    if not config.gradient_mlp:
        raise ValueError("This simulator enforces a gradient MLP (gradient_MLP=True).")
    a = rng.normal(size=(config.n_units, 2))
    a /= np.linalg.norm(a, axis=1, keepdims=True)

    scalars = rng.normal(scale=config.weight_scale, size=(config.n_units, 1))
    omega = scalars * a
    return MLPParams(a=a.astype(np.float64), omega=omega.astype(np.float64), activation=config.activation)



def _attention_drift(
    theta_eval: NDArray[np.float64],
    theta_particles: NDArray[np.float64],
    beta: float,
    mode: AttentionMode,
    self_attention: bool,
    ascending: bool,
) -> NDArray[np.float64]:

    diff = theta_eval[:, None] - theta_particles[None, :]
    sin_diff = np.sin(diff)
    cos_diff = np.cos(diff)

    log_w = beta * (cos_diff - 1.0)  # ≤ 0

    if (not self_attention) and (theta_eval.size == theta_particles.size):
        np.fill_diagonal(log_w, -np.inf)

    weights = np.exp(np.clip(log_w, -80.0, 0.0))  # in [0, 1]

    if (not self_attention) and (theta_eval.size == theta_particles.size):
        np.fill_diagonal(weights, 0.0)

    numerator = (weights * sin_diff).sum(axis=1)
    sign = -1.0 if ascending else 1.0

    if mode == "normalized":
        denom = np.maximum(weights.sum(axis=1), 1e-12)
        return sign * numerator / denom

    return sign * numerator / max(1, theta_particles.size)


def attention_drift_particles(
    theta: NDArray[np.float64],
    beta: float,
    mode: AttentionMode,
    self_attention: bool = False,
    ascending: bool = False,
) -> NDArray[np.float64]:
    """Compute the self-attention drift on S1 (USA/SA model in angle form)."""
    effective_self_attention = self_attention or mode == "normalized"
    return _attention_drift(
        theta,
        theta,
        beta,
        mode,
        effective_self_attention,
        ascending,
    )


def attention_drift_at(
    theta_eval: NDArray[np.float64],
    theta_particles: NDArray[np.float64],
    beta: float,
    mode: AttentionMode,
    ascending: bool = False,
) -> NDArray[np.float64]:
    """Evaluate attention drift at arbitrary angles against a particle set."""
    return _attention_drift(
        theta_eval,
        theta_particles,
        beta,
        mode,
        self_attention=True,
        ascending=ascending,
    )


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
    drift = attention_drift_particles(
        theta,
        sim_config.beta,
        sim_config.attention_mode,
        self_attention=sim_config.self_attention,
        ascending=sim_config.ascending,
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
