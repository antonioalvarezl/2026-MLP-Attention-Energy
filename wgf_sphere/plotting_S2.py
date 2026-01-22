"""Plot utilities for S2 views."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from scipy.special import erf

TWO_PI = 2.0 * np.pi
POINT_ELEVATION = 1.02
POTENTIAL_NEG_COLOR = "#D55E00"
POTENTIAL_ZERO_COLOR = "#E0E0E0"  # Same as neutral sphere
POTENTIAL_POS_COLOR = "#009E73"
POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "potential",
    [POTENTIAL_NEG_COLOR, POTENTIAL_ZERO_COLOR, POTENTIAL_POS_COLOR],
)
POTENTIAL_COLOR_SCALE = [
    [0.0, POTENTIAL_NEG_COLOR],
    [0.5, POTENTIAL_ZERO_COLOR],
    [1.0, POTENTIAL_POS_COLOR],
]

# Decision boundary styling (for histogram floor only)
DECISION_BOUNDARY_COLOR = "#000000"
DECISION_BOUNDARY_LINEWIDTH = 0.6
DECISION_BOUNDARY_ALPHA = 0.85


def _compute_great_circle(normal: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Compute points on the great circle perpendicular to normal."""
    normal = normal / np.linalg.norm(normal)
    if abs(normal[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    u = v - np.dot(v, normal) * normal
    u = u / np.linalg.norm(u)
    w = np.cross(normal, u)
    t = np.linspace(0, TWO_PI, n_points)
    return np.outer(np.cos(t), u) + np.outer(np.sin(t), w)


def get_decision_boundary_circles(
    mlp_params: Optional[tuple[np.ndarray, np.ndarray, str]],
    n_points: int = 100,
) -> list[np.ndarray]:
    """Get great circles for decision boundaries a_i·x = 0."""
    if mlp_params is None:
        return []
    a, omega, activation = mlp_params
    if a.size == 0:
        return []
    circles = []
    for i in range(a.shape[0]):
        normal = a[i]
        if np.linalg.norm(normal) > 1e-10:
            circles.append(_compute_great_circle(normal, n_points))
    return circles


def _primitive_relu(t: np.ndarray) -> np.ndarray:
    return np.maximum(t, 0.0) ** 2


def _primitive_gelu(t: np.ndarray) -> np.ndarray:
    erf_term = erf(t / np.sqrt(2.0))
    return 2.0 * (
        0.5 * t * t
        + (0.5 * t * t - 0.5) * erf_term
        + t * np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)
    )


def mlp_potential(points: np.ndarray, a: np.ndarray, omega: np.ndarray, activation: str) -> np.ndarray:
    if a.size == 0:
        return np.zeros(points.shape[0], dtype=float)
    z = points @ a.T
    if activation == "relu":
        phi = _primitive_relu(z)
    elif activation == "gelu":
        phi = _primitive_gelu(z)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    omega_scalar = (omega * a).sum(axis=1)
    return phi @ omega_scalar


def _potential_on_mesh(
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    mesh_z: np.ndarray,
    params: Optional[tuple[np.ndarray, np.ndarray, str]],
    allow_zero: bool,
) -> Optional[np.ndarray]:
    # When params is None, always return None so we use the neutral sphere
    # (light gray with wireframe grid) instead of a colored surface
    if params is None:
        return None
    a, omega, activation = params
    if a.size == 0:
        return None  # Empty MLP also uses neutral sphere
    points = np.stack([mesh_x, mesh_y, mesh_z], axis=-1).reshape(-1, 3)
    potential = mlp_potential(points, a, omega, activation)
    return potential.reshape(mesh_x.shape)


# --------------------------------------------------------------------------- #
# Cached mesh and potential for GIF generation (static potential optimization)
# --------------------------------------------------------------------------- #

class SphereMeshCache:
    """Pre-computed sphere mesh and potential colors for efficient GIF rendering.
    
    Since the MLP potential is static, we can compute the mesh, potential values,
    and RGBA colors once and reuse them for all frames. This avoids expensive
    MLP evaluation and colormap computation in each frame of the GIF.
    """
    
    def __init__(
        self,
        res_u: int = 240,
        res_v: int = 120,
        null_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
        mlp_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
        show_potential: bool = True,
    ) -> None:
        # Generate sphere mesh once
        self.mesh_x, self.mesh_y, self.mesh_z = _sphere_mesh(res_u=res_u, res_v=res_v)
        
        # Compute potentials
        self.potential_null = _potential_on_mesh(
            self.mesh_x, self.mesh_y, self.mesh_z, null_params, show_potential
        )
        self.potential_mlp = _potential_on_mesh(
            self.mesh_x, self.mesh_y, self.mesh_z, mlp_params, show_potential
        )
        
        # Compute normalization
        max_abs = 0.0
        if self.potential_mlp is not None:
            max_abs = float(np.max(np.abs(self.potential_mlp)))
        elif self.potential_null is not None:
            max_abs = float(np.max(np.abs(self.potential_null)))
        if max_abs <= 0.0:
            max_abs = 1.0
        self.max_abs = max_abs
        self.norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
        
        # Pre-compute RGBA colors for both potentials
        self.colors_null: Optional[np.ndarray] = None
        self.colors_mlp: Optional[np.ndarray] = None
        
        if self.potential_null is not None:
            self.colors_null = POTENTIAL_CMAP(self.norm(self.potential_null))
            self.colors_null[..., 3] = 1.0
        
        if self.potential_mlp is not None:
            self.colors_mlp = POTENTIAL_CMAP(self.norm(self.potential_mlp))
            self.colors_mlp[..., 3] = 1.0


def create_sphere_mesh_cache(
    null_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    mlp_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    show_potential: bool = True,
    res_u: int = 240,
    res_v: int = 120,
) -> SphereMeshCache:
    """Create a cached mesh and potential colors for efficient GIF rendering."""
    return SphereMeshCache(
        res_u=res_u,
        res_v=res_v,
        null_params=null_params,
        mlp_params=mlp_params,
        show_potential=show_potential,
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


def _sphere_mesh(res_u: int = 120, res_v: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, TWO_PI, res_u)
    v = np.linspace(0.0, np.pi, res_v)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _plot_neutral_sphere(ax, mesh_x: np.ndarray, mesh_y: np.ndarray, mesh_z: np.ndarray) -> None:
    """Plot a simple light gray sphere without wireframe or shading."""
    ax.plot_surface(
        mesh_x,
        mesh_y,
        mesh_z,
        rstride=1,
        cstride=1,
        color="#E0E0E0",
        alpha=1.0,
        linewidth=0,
        shade=False,
        antialiased=True,
        zorder=0,
    )


def _style_s2_axis(ax) -> None:
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False


def _lift_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    return points * POINT_ELEVATION


def _view_direction(view_elev: float, view_azim: float) -> np.ndarray:
    elev = np.deg2rad(view_elev)
    azim = np.deg2rad(view_azim)
    return np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=float,
    )


def _filter_points_for_view(
    points: np.ndarray,
    view_elev: float,
    view_azim: float,
) -> np.ndarray:
    if points.size == 0:
        return points
    view = _view_direction(view_elev, view_azim)
    mask = (points @ view) >= 0.0
    if not np.any(mask):
        return points[:0]
    return points[mask]


def _points_to_square_coords(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = np.clip(points[:, 2], -1.0, 1.0)
    azimuth = np.mod(np.arctan2(y, x), TWO_PI)
    polar = np.arccos(z)
    square_v = 2.0 * polar
    square_v = np.where(square_v >= TWO_PI, np.nextafter(TWO_PI, 0.0), square_v)
    return azimuth, square_v


def _plot_decision_boundaries_on_floor(
    ax,
    mlp_params: Optional[tuple[np.ndarray, np.ndarray, str]],
    z_level: float = 0.0,
    n_points: int = 300,
) -> None:
    """Draw decision boundary curves on the histogram floor (φ-θ plane)."""
    circles = get_decision_boundary_circles(mlp_params, n_points)
    
    for circle in circles:
        phi, theta = _points_to_square_coords(circle)
        order = np.argsort(phi)
        phi_sorted = phi[order]
        theta_sorted = theta[order]
        
        # Find discontinuities (jumps > π in phi)
        dphi = np.diff(phi_sorted)
        breaks = np.where(np.abs(dphi) > np.pi)[0] + 1
        
        segments_phi = np.split(phi_sorted, breaks)
        segments_theta = np.split(theta_sorted, breaks)
        
        for seg_phi, seg_theta in zip(segments_phi, segments_theta):
            if len(seg_phi) > 1:
                ax.plot(
                    seg_phi,
                    seg_theta,
                    np.full_like(seg_phi, z_level),
                    color=DECISION_BOUNDARY_COLOR,
                    linewidth=DECISION_BOUNDARY_LINEWIDTH,
                    alpha=DECISION_BOUNDARY_ALPHA,
                    zorder=1.5,
                )


def _potential_on_square_grid(
    phi_grid: np.ndarray,
    theta_grid: np.ndarray,
    params: Optional[tuple[np.ndarray, np.ndarray, str]],
    show_potential: bool,
) -> Optional[np.ndarray]:
    """Compute MLP potential on a square grid representing the sphere.
    
    The grid uses coordinates (phi, theta) where:
    - phi in [0, 2π] is the azimuthal angle
    - theta in [0, 2π] maps to polar angle via polar = theta/2, so theta in [0, 2π] -> polar in [0, π]
    
    This function ensures proper continuity:
    - At theta=0 (north pole) and theta=2π (south pole), all phi values map to the same point,
      so we use a constant potential value (the average over phi at that polar angle).
    - At phi=0 and phi=2π, the values are identical (periodic boundary).
    """
    if params is None or not show_potential:
        return None
    a, omega, activation = params
    phi, theta = np.meshgrid(phi_grid, theta_grid, indexing="ij")
    polar = 0.5 * theta
    sin_polar = np.sin(polar)
    x = np.cos(phi) * sin_polar
    y = np.sin(phi) * sin_polar
    z = np.cos(polar)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    potential = mlp_potential(points, a, omega, activation)
    potential = potential.reshape(phi.shape)
    
    # Fix polar singularities: at theta=0 (north pole) and theta=2π (south pole),
    # all azimuthal angles map to the same physical point, so the potential should
    # be constant across phi. We take the mean to ensure smooth visualization.
    # Check for theta values very close to 0 or 2π (poles)
    eps = 1e-6
    north_pole_mask = theta_grid < eps
    south_pole_mask = theta_grid > (TWO_PI - eps)
    
    if np.any(north_pole_mask):
        # All columns at theta≈0 should have the same value
        for j, is_pole in enumerate(north_pole_mask):
            if is_pole:
                pole_value = potential[:, j].mean()
                potential[:, j] = pole_value
    
    if np.any(south_pole_mask):
        # All columns at theta≈2π should have the same value
        for j, is_pole in enumerate(south_pole_mask):
            if is_pole:
                pole_value = potential[:, j].mean()
                potential[:, j] = pole_value
    
    return potential


def _slerp_points(p0: np.ndarray, p1: np.ndarray, alpha: float) -> np.ndarray:
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


def _resample_trajectory(
    points_hist: np.ndarray,
    times: np.ndarray,
    target_steps: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    steps = min(points_hist.shape[0], times.shape[0])
    if steps < 2 or target_steps <= 1 or target_steps >= steps:
        return points_hist[:steps], times[:steps]
    times = np.asarray(times[:steps], dtype=float)
    target_steps = int(target_steps)
    target_times = np.linspace(times[0], times[-1], target_steps)
    idx = np.searchsorted(times, target_times, side="right") - 1
    idx = np.clip(idx, 0, steps - 2)

    dense_points = np.empty((target_steps, points_hist.shape[1], points_hist.shape[2]), dtype=float)
    for out_idx, base_idx in enumerate(idx):
        t0 = float(times[base_idx])
        t1 = float(times[base_idx + 1])
        if t1 <= t0:
            dense_points[out_idx] = points_hist[base_idx]
        else:
            alpha = (target_times[out_idx] - t0) / (t1 - t0)
            dense_points[out_idx] = _slerp_points(
                points_hist[base_idx],
                points_hist[base_idx + 1],
                float(alpha),
            )
    return dense_points, target_times


def _plot_s2_trajectories(
    ax,
    points_hist: np.ndarray,
    times: np.ndarray,
    color: str,
    line_width: float = 0.6,
    alpha: float = 0.65,
    z_aspect: float = 2.2,
    target_steps: int = 3000,
    time_scale: str = "linear",
) -> None:
    """Plot particle trajectories on S2 projected to square coordinates.
    
    Args:
        time_scale: "linear" or "log" for the time axis.
    """
    times = np.asarray(times, dtype=float)
    if points_hist.size == 0 or times.size == 0:
        return
    steps = min(points_hist.shape[0], times.shape[0])
    if steps < 2:
        return
    points_hist = points_hist[:steps]
    times = times[:steps]
    points_hist, times = _resample_trajectory(
        points_hist,
        times,
        target_steps=target_steps,
    )
    steps = min(points_hist.shape[0], times.shape[0])
    if steps < 2:
        return

    phi = np.empty((steps, points_hist.shape[1]), dtype=float)
    theta = np.empty_like(phi)
    for idx in range(steps):
        phi[idx], theta[idx] = _points_to_square_coords(points_hist[idx])

    # Prepare time axis (log or linear)
    t_min_orig = float(times[0])
    t_max_orig = float(times[-1])
    
    if time_scale == "log":
        # Transform times to log scale (matplotlib 3D doesn't support set_zscale('log'))
        t_min = max(t_min_orig, 1e-6)
        times_safe = np.maximum(times, t_min)
        plot_times = np.log10(times_safe)
        z_min = np.log10(t_min)
        z_max = np.log10(t_max_orig)
    else:
        plot_times = times
        z_min = t_min_orig
        z_max = t_max_orig

    jump_thresh = np.pi
    for p_idx in range(phi.shape[1]):
        phi_p = phi[:, p_idx]
        theta_p = theta[:, p_idx]
        jumps = (np.abs(np.diff(phi_p)) > jump_thresh) | (np.abs(np.diff(theta_p)) > jump_thresh)
        start = 0
        for idx, is_jump in enumerate(jumps):
            if is_jump:
                if idx + 1 - start >= 2:
                    ax.plot(
                        phi_p[start : idx + 1],
                        theta_p[start : idx + 1],
                        plot_times[start : idx + 1],
                        color=color,
                        linewidth=line_width,
                        alpha=alpha,
                    )
                start = idx + 1
        if steps - start >= 2:
            ax.plot(
                phi_p[start:],
                theta_p[start:],
                plot_times[start:],
                color=color,
                linewidth=line_width,
                alpha=alpha,
            )

    ticks = [0.0, np.pi, TWO_PI]
    tick_labels = [r"$0$", r"$\pi$", r"$2\pi$"]
    ax.set_xlim(0.0, TWO_PI)
    ax.set_ylim(0.0, TWO_PI)
    ax.set_zlim(z_min, z_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    
    # Set z-axis ticks
    if time_scale == "log":
        z_tick_values = np.array([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        t_min_safe = max(t_min_orig, 1e-6)
        z_tick_values = z_tick_values[(z_tick_values >= t_min_safe * 0.5) & (z_tick_values <= t_max_orig * 2)]
        if len(z_tick_values) < 2:
            z_tick_values = np.array([t_min_safe, t_max_orig])
        z_tick_log = np.log10(z_tick_values)
        z_tick_labels = [f"{v:g}" for v in z_tick_values]
        ax.set_zticks(z_tick_log)
        ax.set_zticklabels(z_tick_labels, fontsize=9)
    else:
        # Linear scale - let matplotlib choose ticks automatically
        ax.tick_params(axis='z', labelsize=9)
    
    ax.set_xlabel(r"$\phi$", fontsize=11, labelpad=12)
    ax.set_ylabel(r"$\theta$", fontsize=11, labelpad=12)
    ax.set_zlabel(r"$t$", fontsize=11, labelpad=10)
    ax.tick_params(axis='x', pad=6)
    ax.tick_params(axis='y', pad=6)
    ax.tick_params(axis='z', pad=8)
    ax.view_init(elev=28, azim=-60)
    ax.grid(False)
    ax.set_box_aspect((1.0, 1.0, z_aspect))


def make_s2_trajectory_figure(
    points_hist: np.ndarray,
    times: np.ndarray,
    color: str,
    z_aspect: float = 2.2,
    target_steps: int = 3000,
    time_scale: str = "linear",
) -> "plt.Figure":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d

    fig = plt.figure(figsize=(8.0, 10.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _plot_s2_trajectories(
        ax,
        points_hist,
        times,
        color,
        z_aspect=z_aspect,
        target_steps=target_steps,
        time_scale=time_scale,
    )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    return fig


def make_s2_trajectory_comparison_figure(
    points_null_hist: np.ndarray,
    times_null: np.ndarray,
    points_mlp_hist: np.ndarray,
    times_mlp: np.ndarray,
    null_color: str,
    mlp_color: str,
    title: str = "",
    time_scale: str = "linear",
) -> "plt.Figure":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d

    fig = plt.figure(figsize=(14.0, 7.0))
    ax_null = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mlp = fig.add_subplot(1, 2, 2, projection="3d")

    _plot_s2_trajectories(ax_null, points_null_hist, times_null, null_color, z_aspect=1.4, time_scale=time_scale)
    _plot_s2_trajectories(ax_mlp, points_mlp_hist, times_mlp, mlp_color, z_aspect=1.4, time_scale=time_scale)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, wspace=0.16)
    return fig


def s2_histogram_max_count(points: np.ndarray, bins: int = 36) -> float:
    if points.size == 0:
        return 0.0
    azimuth, square_v = _points_to_square_coords(points)
    counts, _, _ = np.histogram2d(
        azimuth,
        square_v,
        bins=bins,
        range=((0.0, TWO_PI), (0.0, TWO_PI)),
    )
    return float(np.max(counts)) if counts.size else 0.0


def _plot_s2_histogram_bars(
    ax,
    points: np.ndarray,
    color: str,
    bins: int,
    potential_params: Optional[tuple[np.ndarray, np.ndarray, str]],
    show_potential: bool,
    z_max: Optional[float],
    show_decision_boundaries: bool = False,
) -> None:
    from matplotlib.ticker import MaxNLocator

    azimuth, square_v = _points_to_square_coords(points)
    counts, x_edges, y_edges = np.histogram2d(
        azimuth,
        square_v,
        bins=bins,
        range=((0.0, TWO_PI), (0.0, TWO_PI)),
    )
    x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
    x_pos = x_pos.ravel()
    y_pos = y_pos.ravel()
    z_pos = np.zeros_like(x_pos)
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    dz = counts.ravel()

    max_count = float(dz.max()) if dz.size else 0.0
    if z_max is None:
        z_limit = max(1.0, max_count)
    else:
        z_limit = max(1.0, float(z_max))
    base_z = 0.0

    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    floor_res = max(600, bins * 12)
    phi_floor = np.linspace(0.0, TWO_PI, floor_res)
    theta_floor = np.linspace(0.0, TWO_PI, floor_res)
    floor_potential = _potential_on_square_grid(
        phi_floor, theta_floor, potential_params, show_potential
    )
    phi_mesh, theta_mesh = np.meshgrid(phi_floor, theta_floor, indexing="ij")
    if floor_potential is None:
        floor_rgba = to_rgba("#DDDDDD", alpha=1.0)
        colors = np.empty(phi_mesh.shape + (4,), dtype=float)
        colors[:] = floor_rgba
        floor_surface = ax.plot_surface(
            phi_mesh,
            theta_mesh,
            np.full_like(phi_mesh, base_z),
            facecolors=colors,
            linewidth=0,
            shade=False,
            antialiased=True,
        )
    else:
        max_abs = float(np.max(np.abs(floor_potential)))
        if max_abs <= 0.0:
            max_abs = 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
        colors = POTENTIAL_CMAP(norm(floor_potential))
        colors[..., 3] = 1.0
        floor_surface = ax.plot_surface(
            phi_mesh,
            theta_mesh,
            np.full_like(floor_potential, base_z),
            facecolors=colors,
            linewidth=0,
            shade=False,
            antialiased=True,
        )
    floor_surface.set_zorder(0)
    floor_surface.set_zsort("min")

    mask = dz > 0.0
    if np.any(mask):
        bars = ax.bar3d(
            x_pos[mask],
            y_pos[mask],
            z_pos[mask],
            dx,
            dy,
            dz[mask],
            color=color,
            alpha=0.94,
            shade=False,
            linewidth=0.25,
            edgecolor="#000000",
        )
        bars.set_zsort("max")
        bars.set_zorder(2)
    shadow_alpha = 0.25
    shadow_color = "#444444"
    shadow_depth = 0.15 * dy
    shadow_width = 0.15 * dx
    shadow_x = counts.max(axis=1) if counts.size else np.zeros_like(x_edges[:-1])
    shadow_y = counts.max(axis=0) if counts.size else np.zeros_like(y_edges[:-1])
    shadow_xbar = ax.bar3d(
        x_edges[:-1],
        np.full_like(x_edges[:-1], TWO_PI - shadow_depth),
        np.full_like(shadow_x, base_z),
        dx,
        shadow_depth,
        shadow_x,
        color=shadow_color,
        alpha=shadow_alpha,
        shade=False,
        linewidth=0.0,
    )
    shadow_ybar = ax.bar3d(
        np.full_like(y_edges[:-1], TWO_PI - shadow_width),
        y_edges[:-1],
        np.full_like(shadow_y, base_z),
        shadow_width,
        dy,
        shadow_y,
        color=shadow_color,
        alpha=shadow_alpha,
        shade=False,
        linewidth=0.0,
    )
    if shadow_xbar is not None:
        shadow_xbar.set_zorder(1)
        shadow_xbar.set_zsort("min")
    if shadow_ybar is not None:
        shadow_ybar.set_zorder(1)
        shadow_ybar.set_zsort("min")
    
    # Draw decision boundaries on the floor if requested
    if show_decision_boundaries and potential_params is not None:
        _plot_decision_boundaries_on_floor(ax, potential_params, z_level=base_z + 0.01)
    
    ticks = [0.0, np.pi, TWO_PI]
    tick_labels = [r"$0$", r"$\pi$", r"$2\pi$"]
    ax.set_xlim(0.0, TWO_PI)
    ax.set_ylim(0.0, TWO_PI)
    ax.set_zlim(0.0, z_limit)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
    ax.set_xlabel(r"$\phi$", labelpad=10)
    ax.set_ylabel(r"$\theta$", labelpad=10)
    ax.set_zlabel("count", labelpad=8)
    ax.tick_params(axis='x', pad=4)
    ax.tick_params(axis='y', pad=4)
    ax.tick_params(axis='z', pad=6)
    ax.view_init(elev=25, azim=-135)
    ax.grid(False)


def make_s2_histogram_bar_figure(
    points: np.ndarray,
    color: str,
    bins: int = 36,
    potential_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    show_potential: bool = True,
    z_max: Optional[float] = None,
    title: str = "",
    show_decision_boundaries: bool = False,
) -> "plt.Figure":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d
    fig = plt.figure(figsize=(6.4, 5.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _plot_s2_histogram_bars(
        ax,
        points,
        color,
        bins,
        potential_params,
        show_potential,
        z_max,
        show_decision_boundaries,
    )
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.86)
    else:
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.96)
    return fig


def make_s2_histogram_comparison_figure(
    points_null: np.ndarray,
    points_mlp: np.ndarray,
    null_color: str,
    mlp_color: str,
    title: str = "",
    bins: int = 36,
    null_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    mlp_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    show_null_potential: bool = False,
    show_mlp_potential: bool = True,
    z_max: Optional[float] = None,
    show_decision_boundaries: bool = False,
) -> "plt.Figure":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d

    if z_max is None:
        z_max = max(
            s2_histogram_max_count(points_null, bins=bins),
            s2_histogram_max_count(points_mlp, bins=bins),
        )
    fig = plt.figure(figsize=(11.2, 6.0))
    ax_null = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mlp = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_s2_histogram_bars(
        ax_null,
        points_null,
        null_color,
        bins,
        null_params,
        show_null_potential,
        z_max,
        show_decision_boundaries=False,  # Never on null
    )
    _plot_s2_histogram_bars(
        ax_mlp,
        points_mlp,
        mlp_color,
        bins,
        mlp_params,
        show_mlp_potential,
        z_max,
        show_decision_boundaries,  # Only on MLP side if requested
    )
    if title:
        fig.suptitle(title)
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.84, wspace=0.18)
    else:
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.96, wspace=0.18)
    return fig


def make_s2_single_figure(
    points: np.ndarray,
    color: str,
    potential_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    show_potential: bool = True,
    point_size: float = 18.0,
    view_elev: float = 20.0,
    view_azim: float = 35.0,
    clip_back: bool = True,
) -> "plt.Figure":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d

    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    mesh_x, mesh_y, mesh_z = _sphere_mesh(res_u=240, res_v=120)
    potential = _potential_on_mesh(mesh_x, mesh_y, mesh_z, potential_params, show_potential)
    max_abs = 0.0
    if potential is not None:
        max_abs = float(np.max(np.abs(potential)))
    if max_abs <= 0.0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

    if potential is None:
        _plot_neutral_sphere(ax, mesh_x, mesh_y, mesh_z)
    else:
        colors = POTENTIAL_CMAP(norm(potential))
        colors[..., 3] = 1.0
        ax.plot_surface(
            mesh_x,
            mesh_y,
            mesh_z,
            rstride=1,
            cstride=1,
            facecolors=colors,
            linewidth=0,
            shade=False,
            antialiased=True,
            zorder=0,
        )
    
    plot_points = _filter_points_for_view(points, view_elev, view_azim) if clip_back else points
    lifted = _lift_points(plot_points)
    ax.scatter(
        lifted[:, 0],
        lifted[:, 1],
        lifted[:, 2],
        s=point_size,
        color=color,
        alpha=0.9,
        edgecolors="none",
        linewidths=0.0,
        depthshade=False,
        zorder=1,
    )
    _style_s2_axis(ax)
    ax.view_init(elev=view_elev, azim=view_azim)

    fig.suptitle("")
    fig.tight_layout()
    return fig


def make_mlp_potential_surface_figure(
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    res_u: int = 240,
    res_v: int = 120,
    view_elev: float = 20.0,
    view_azim: float = 35.0,
) -> "plt.Figure":
    mesh_x, mesh_y, mesh_z = _sphere_mesh(res_u=res_u, res_v=res_v)
    points = np.stack([mesh_x, mesh_y, mesh_z], axis=-1).reshape(-1, 3)
    potential = mlp_potential(points, a, omega, activation).reshape(mesh_x.shape)
    max_abs = float(np.max(np.abs(potential))) if potential.size else 0.0
    if max_abs <= 0.0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    colors = POTENTIAL_CMAP(norm(potential))
    colors[..., 3] = 1.0

    fig = plt.figure(figsize=(5.5, 5.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(
        mesh_x,
        mesh_y,
        mesh_z,
        rstride=1,
        cstride=1,
        facecolors=colors,
        linewidth=0,
        shade=False,
        antialiased=True,
        zorder=0,
    )
    _style_s2_axis(ax)
    ax.view_init(elev=view_elev, azim=view_azim)
    fig.tight_layout()
    return fig


def make_s2_comparison_figure(
    points_null: np.ndarray,
    points_mlp: np.ndarray,
    null_color: str,
    mlp_color: str,
    title: str,
    null_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    mlp_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    show_potential: bool = True,
    point_size: float = 18.0,
    view_elev: float = 20.0,
    view_azim: float = 35.0,
    show_title: bool = False,
    clip_back: bool = True,
    mesh_cache: Optional[SphereMeshCache] = None,
) -> "plt.Figure":
    """Create a comparison figure with two spheres (null and MLP).
    
    If mesh_cache is provided, uses pre-computed mesh and potential colors
    for much faster rendering (useful for GIF generation with static potential).
    """
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d

    fig = plt.figure(figsize=(9.0, 4.5))
    ax_null = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mlp = fig.add_subplot(1, 2, 2, projection="3d")

    # Use cached mesh and colors if available, otherwise compute on the fly
    if mesh_cache is not None:
        mesh_x = mesh_cache.mesh_x
        mesh_y = mesh_cache.mesh_y
        mesh_z = mesh_cache.mesh_z
        colors_null = mesh_cache.colors_null
        colors_mlp = mesh_cache.colors_mlp
        potential_null = mesh_cache.potential_null
        potential_mlp = mesh_cache.potential_mlp
    else:
        mesh_x, mesh_y, mesh_z = _sphere_mesh(res_u=240, res_v=120)
        potential_null = _potential_on_mesh(mesh_x, mesh_y, mesh_z, null_params, show_potential)
        potential_mlp = _potential_on_mesh(mesh_x, mesh_y, mesh_z, mlp_params, show_potential)
        max_abs = 0.0
        if potential_mlp is not None:
            max_abs = float(np.max(np.abs(potential_mlp)))
        elif potential_null is not None:
            max_abs = float(np.max(np.abs(potential_null)))
        if max_abs <= 0.0:
            max_abs = 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
        
        colors_null = None
        colors_mlp = None
        if potential_null is not None:
            colors_null = POTENTIAL_CMAP(norm(potential_null))
            colors_null[..., 3] = 1.0
        if potential_mlp is not None:
            colors_mlp = POTENTIAL_CMAP(norm(potential_mlp))
            colors_mlp[..., 3] = 1.0

    for ax, points, color, label, potential, facecolors in (
        (ax_null, points_null, null_color, "MLP=0", potential_null, colors_null),
        (ax_mlp, points_mlp, mlp_color, "MLP", potential_mlp, colors_mlp),
    ):
        if potential is None:
            _plot_neutral_sphere(ax, mesh_x, mesh_y, mesh_z)
        else:
            ax.plot_surface(
                mesh_x,
                mesh_y,
                mesh_z,
                rstride=1,
                cstride=1,
                facecolors=facecolors,
                linewidth=0,
                shade=False,
                antialiased=True,
                zorder=0,
            )
        
        plot_points = (
            _filter_points_for_view(points, view_elev, view_azim) if clip_back else points
        )
        lifted = _lift_points(plot_points)
        ax.scatter(
            lifted[:, 0],
            lifted[:, 1],
            lifted[:, 2],
            s=point_size,
            color=color,
            alpha=0.9,
            edgecolors="none",
            linewidths=0.0,
            depthshade=False,
            zorder=1,
        )
        _style_s2_axis(ax)
        ax.view_init(elev=view_elev, azim=view_azim)

    if show_title and title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    else:
        fig.suptitle("")
        fig.tight_layout()
    return fig


def write_s2_interactive_html(
    output_path: Path,
    initial: np.ndarray,
    middle: np.ndarray,
    final: np.ndarray,
    color: str,
    mlp_params: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
    show_potential: bool = True,
    point_size: float = 10.0,
    # Data for null case (MLP=0)
    initial_null: Optional[np.ndarray] = None,
    middle_null: Optional[np.ndarray] = None,
    final_null: Optional[np.ndarray] = None,
    null_color: Optional[str] = None,
    histogram_bins: int = 18,
) -> None:
    """Write interactive HTML with sphere visualization.
    
    Features:
    - Toggle between MLP=0 (null) and MLP views
    - Toggle between Points and Histogram display modes
    - Select time state: Initial, Middle, Final
    """
    mesh_x, mesh_y, mesh_z = _sphere_mesh()
    
    # Use same data for null if not provided
    if initial_null is None:
        initial_null = initial
    if middle_null is None:
        middle_null = middle
    if final_null is None:
        final_null = final
    if null_color is None:
        null_color = "#7F7F7F"  # Default gray for null
    
    # Compute potential for MLP sphere
    potential_mlp = _potential_on_mesh(mesh_x, mesh_y, mesh_z, mlp_params, show_potential)
    max_abs = 0.0
    if potential_mlp is not None:
        max_abs = float(np.max(np.abs(potential_mlp)))
    if max_abs <= 0.0:
        max_abs = 1.0

    def _sphere_trace_neutral() -> dict:
        """Neutral gray sphere for MLP=0."""
        return {
            "type": "surface",
            "x": mesh_x.tolist(),
            "y": mesh_y.tolist(),
            "z": mesh_z.tolist(),
            "opacity": 1.0,
            "showscale": False,
            "hoverinfo": "skip",
            "colorscale": [[0.0, POTENTIAL_ZERO_COLOR], [1.0, POTENTIAL_ZERO_COLOR]],
            "lighting": {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0},
        }

    def _sphere_trace_potential() -> dict:
        """Sphere with potential coloring for MLP."""
        if potential_mlp is None:
            return _sphere_trace_neutral()
        return {
            "type": "surface",
            "x": mesh_x.tolist(),
            "y": mesh_y.tolist(),
            "z": mesh_z.tolist(),
            "surfacecolor": potential_mlp.tolist(),
            "colorscale": POTENTIAL_COLOR_SCALE,
            "cmin": -max_abs,
            "cmax": max_abs,
            "opacity": 1.0,
            "showscale": False,
            "hoverinfo": "skip",
            "lighting": {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0},
        }

    def _scatter_trace(points: np.ndarray, pt_color: str) -> dict:
        """Scatter points on sphere."""
        lifted = _lift_points(points)
        return {
            "type": "scatter3d",
            "mode": "markers",
            "x": lifted[:, 0].tolist(),
            "y": lifted[:, 1].tolist(),
            "z": lifted[:, 2].tolist(),
            "marker": {
                "size": point_size,
                "color": pt_color,
                "opacity": 0.9,
                "line": {"color": "#000000", "width": 0.6},
            },
            "showlegend": False,
            "hoverinfo": "skip",
        }
    
    def _histogram_bars(points: np.ndarray, pt_color: str) -> list:
        """Create 3D cylinders on sphere surface representing histogram bins.
        
        Each bin is shown as a cylinder with:
        - Radius proportional to sqrt(count) for area scaling
        - Height proportional to count
        - Positioned at the bin center on the sphere surface
        """
        if points.size == 0:
            return []
        
        # Compute histogram on sphere using theta/phi bins
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        z_clipped = np.clip(z, -1.0, 1.0)
        phi = np.mod(np.arctan2(y, x), TWO_PI)
        theta = np.arccos(z_clipped)
        
        # Higher resolution: more bins for better visual
        n_phi_bins = histogram_bins
        n_theta_bins = max(histogram_bins // 2, 9)
        
        phi_edges = np.linspace(0, TWO_PI, n_phi_bins + 1)
        theta_edges = np.linspace(0, np.pi, n_theta_bins + 1)
        counts, _, _ = np.histogram2d(phi, theta, bins=[phi_edges, theta_edges])
        
        max_count = counts.max() if counts.size else 1.0
        if max_count <= 0:
            max_count = 1.0
        
        # Cylinder parameters - keep small to not distort sphere
        cylinder_max_height = 0.15  # Maximum height
        cylinder_max_radius = 0.03  # Maximum radius
        n_circle_pts = 8  # Points per cylinder circle
        
        bar_traces = []
        
        for i in range(len(phi_edges) - 1):
            for j in range(len(theta_edges) - 1):
                count = counts[i, j]
                if count <= 0:
                    continue
                
                # Center of bin
                phi_c = (phi_edges[i] + phi_edges[i + 1]) / 2
                theta_c = (theta_edges[j] + theta_edges[j + 1]) / 2
                
                # Position on sphere (base of cylinder)
                sin_theta = np.sin(theta_c)
                base_x = np.cos(phi_c) * sin_theta
                base_y = np.sin(phi_c) * sin_theta
                base_z = np.cos(theta_c)
                normal = np.array([base_x, base_y, base_z])
                
                # Cylinder dimensions based on count
                fraction = count / max_count
                height = cylinder_max_height * fraction
                radius = cylinder_max_radius * np.sqrt(fraction)  # Area scales with count
                
                # Skip very small cylinders
                if height < 0.01:
                    continue
                
                # Build cylinder mesh: two circles connected
                # Create orthonormal basis perpendicular to normal
                if abs(normal[0]) < 0.9:
                    v = np.array([1.0, 0.0, 0.0])
                else:
                    v = np.array([0.0, 1.0, 0.0])
                u1 = v - np.dot(v, normal) * normal
                u1 = u1 / np.linalg.norm(u1)
                u2 = np.cross(normal, u1)
                
                # Circle points
                angles = np.linspace(0, TWO_PI, n_circle_pts, endpoint=False)
                circle_local = np.zeros((n_circle_pts, 2))
                circle_local[:, 0] = np.cos(angles)
                circle_local[:, 1] = np.sin(angles)
                
                # Bottom circle (on sphere surface, slightly elevated)
                base_elevation = 1.003
                bottom_center = normal * base_elevation
                bottom = np.zeros((n_circle_pts, 3))
                for k in range(n_circle_pts):
                    bottom[k] = bottom_center + radius * (circle_local[k, 0] * u1 + circle_local[k, 1] * u2)
                
                # Top circle
                top_center = normal * (base_elevation + height)
                top = np.zeros((n_circle_pts, 3))
                for k in range(n_circle_pts):
                    top[k] = top_center + radius * (circle_local[k, 0] * u1 + circle_local[k, 1] * u2)
                
                # Create mesh3d vertices and faces for cylinder
                # Vertices: bottom circle (0 to n-1), top circle (n to 2n-1), bottom center (2n), top center (2n+1)
                verts_x = list(bottom[:, 0]) + list(top[:, 0]) + [bottom_center[0], top_center[0]]
                verts_y = list(bottom[:, 1]) + list(top[:, 1]) + [bottom_center[1], top_center[1]]
                verts_z = list(bottom[:, 2]) + list(top[:, 2]) + [bottom_center[2], top_center[2]]
                
                n = n_circle_pts
                faces_i = []
                faces_j = []
                faces_k = []
                
                # Side faces (quads as two triangles)
                for k in range(n):
                    k_next = (k + 1) % n
                    # Triangle 1
                    faces_i.append(k)
                    faces_j.append(k_next)
                    faces_k.append(n + k)
                    # Triangle 2
                    faces_i.append(k_next)
                    faces_j.append(n + k_next)
                    faces_k.append(n + k)
                
                # Bottom cap
                bottom_center_idx = 2 * n
                for k in range(n):
                    k_next = (k + 1) % n
                    faces_i.append(bottom_center_idx)
                    faces_j.append(k_next)
                    faces_k.append(k)
                
                # Top cap
                top_center_idx = 2 * n + 1
                for k in range(n):
                    k_next = (k + 1) % n
                    faces_i.append(top_center_idx)
                    faces_j.append(n + k)
                    faces_k.append(n + k_next)
                
                bar_traces.append({
                    "type": "mesh3d",
                    "x": verts_x,
                    "y": verts_y,
                    "z": verts_z,
                    "i": faces_i,
                    "j": faces_j,
                    "k": faces_k,
                    "color": pt_color,
                    "opacity": 0.9,
                    "flatshading": True,
                    "showscale": False,
                    "hoverinfo": "skip",
                    "lighting": {
                        "ambient": 0.6,
                        "diffuse": 0.5,
                        "specular": 0.2,
                    },
                })
        
        return bar_traces

    # Pre-compute all data
    axis = {
        "visible": False,
        "showgrid": False,
        "zeroline": False,
        "showbackground": False,
    }
    
    # Store all state data
    states_mlp = {
        "Initial": {"points": initial, "color": color},
        "Middle": {"points": middle, "color": color},
        "Final": {"points": final, "color": color},
    }
    states_null = {
        "Initial": {"points": initial_null, "color": null_color},
        "Middle": {"points": middle_null, "color": null_color},
        "Final": {"points": final_null, "color": null_color},
    }
    
    # Serialize data for JavaScript
    def pts_to_json(pts):
        lifted = _lift_points(pts)
        return {
            "x": lifted[:, 0].tolist(),
            "y": lifted[:, 1].tolist(),
            "z": lifted[:, 2].tolist(),
        }
    
    def hist_to_json(pts, pt_color):
        bars = _histogram_bars(pts, pt_color)
        return bars
    
    # Create initial traces (MLP + points by default)
    data = [
        _sphere_trace_potential(),
        _scatter_trace(initial, color),
    ]
    
    layout = {
        "margin": {"l": 20, "r": 20, "t": 20, "b": 20},
        "paper_bgcolor": "white",
        "scene": {
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 1},
            "xaxis": {**axis, "range": [-1.4, 1.4], "autorange": False},
            "yaxis": {**axis, "range": [-1.4, 1.4], "autorange": False},
            "zaxis": {**axis, "range": [-1.4, 1.4], "autorange": False},
            "dragmode": "orbit",
            "camera": {
                "eye": {"x": 1.5, "y": 1.5, "z": 1.2},
            },
        },
    }
    
    # Prepare all data for JavaScript
    js_data = {
        "sphere_null": _sphere_trace_neutral(),
        "sphere_mlp": _sphere_trace_potential(),
        "states_mlp_points": {k: pts_to_json(v["points"]) for k, v in states_mlp.items()},
        "states_null_points": {k: pts_to_json(v["points"]) for k, v in states_null.items()},
        "states_mlp_hist": {k: hist_to_json(v["points"], v["color"]) for k, v in states_mlp.items()},
        "states_null_hist": {k: hist_to_json(v["points"], v["color"]) for k, v in states_null.items()},
        "color_mlp": color,
        "color_null": null_color,
        "point_size": point_size,
    }

    payload = {
        "data": data,
        "layout": layout,
        "config": {"responsive": True, "displaylogo": False},
    }
    
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>S2 Interactive View</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js"></script>
  <style>
    body {{ margin: 0; background: #ffffff; font-family: Arial, sans-serif; }}
    #controls {{
      display: flex;
      gap: 16px;
      padding: 10px 16px;
      align-items: center;
      flex-wrap: wrap;
      background: #f5f5f5;
      border-bottom: 1px solid #ddd;
    }}
    .control-group {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .control-group label {{
      font-size: 13px;
      font-weight: 500;
      color: #333;
    }}
    select, button {{
      font-size: 13px;
      padding: 6px 12px;
      border: 1px solid #ccc;
      border-radius: 4px;
      background: white;
      cursor: pointer;
    }}
    select:hover, button:hover {{
      border-color: #999;
    }}
    button {{
      background: #4CAF50;
      color: white;
      border: none;
    }}
    button:hover {{
      background: #45a049;
    }}
    #plot {{ width: 100%; height: calc(100vh - 60px); }}
  </style>
</head>
<body>
  <div id="controls">
    <div class="control-group">
      <label>Model:</label>
      <select id="model-select">
        <option value="mlp" selected>MLP</option>
        <option value="null">MLP=0</option>
      </select>
    </div>
    <div class="control-group">
      <label>Display:</label>
      <select id="display-select">
        <option value="points" selected>Points</option>
        <option value="histogram">Histogram</option>
      </select>
    </div>
    <div class="control-group">
      <label>Time:</label>
      <select id="time-select">
        <option value="Initial" selected>Initial</option>
        <option value="Middle">Middle</option>
        <option value="Final">Final</option>
      </select>
    </div>
    <button id="save-pdf">Save PDF</button>
    <button id="save-png">Save PNG</button>
  </div>
  <div id="plot"></div>
  <script>
    const initialPayload = {json.dumps(payload, ensure_ascii=True, separators=(",", ":"))};
    const jsData = {json.dumps(js_data, ensure_ascii=True, separators=(",", ":"))};
    
    let currentModel = "mlp";
    let currentDisplay = "points";
    let currentTime = "Initial";
    
    Plotly.newPlot("plot", initialPayload.data, initialPayload.layout, initialPayload.config);
    
    function updatePlot() {{
      const plotEl = document.getElementById("plot");
      
      // Select sphere trace
      const sphereTrace = currentModel === "mlp" ? jsData.sphere_mlp : jsData.sphere_null;
      
      // Select points/histogram data
      let dataTraces = [sphereTrace];
      
      if (currentDisplay === "points") {{
        const statesKey = currentModel === "mlp" ? "states_mlp_points" : "states_null_points";
        const pts = jsData[statesKey][currentTime];
        const ptColor = currentModel === "mlp" ? jsData.color_mlp : jsData.color_null;
        dataTraces.push({{
          type: "scatter3d",
          mode: "markers",
          x: pts.x,
          y: pts.y,
          z: pts.z,
          marker: {{
            size: jsData.point_size,
            color: ptColor,
            opacity: 0.9,
            line: {{ color: "#000000", width: 0.6 }},
          }},
          showlegend: false,
          hoverinfo: "skip",
        }});
      }} else {{
        // Histogram mode - add cylinder traces
        const histKey = currentModel === "mlp" ? "states_mlp_hist" : "states_null_hist";
        const cylinders = jsData[histKey][currentTime];
        dataTraces = dataTraces.concat(cylinders);
      }}
      
      Plotly.react("plot", dataTraces, initialPayload.layout, initialPayload.config);
    }}
    
    document.getElementById("model-select").addEventListener("change", function(e) {{
      currentModel = e.target.value;
      updatePlot();
    }});
    
    document.getElementById("display-select").addEventListener("change", function(e) {{
      currentDisplay = e.target.value;
      updatePlot();
    }});
    
    document.getElementById("time-select").addEventListener("change", function(e) {{
      currentTime = e.target.value;
      updatePlot();
    }});
    
    document.getElementById("save-pdf").addEventListener("click", function() {{
      const plotEl = document.getElementById("plot");
      const controls = document.getElementById("controls");
      
      // Hide controls and make plot full screen for capture
      controls.style.display = "none";
      plotEl.style.height = "100vh";
      
      // High resolution for quality output (2400x2400 pixels)
      const size = 2400;
      const scale = 3;  // Higher scale for better quality
      
      // Small delay to let layout update
      setTimeout(function() {{
        Plotly.toImage(plotEl, {{ format: "png", width: size, height: size, scale: scale }}).then(function(dataUrl) {{
          const jsPDF = window.jspdf.jsPDF;
          // PDF at 300 DPI: 2400px / 300dpi * 72pt/inch = 576pt
          const pdfSize = 576;
          const doc = new jsPDF({{ unit: "pt", format: [pdfSize, pdfSize] }});
          doc.addImage(dataUrl, "PNG", 0, 0, pdfSize, pdfSize);
          doc.save("sphere_view.pdf");
          
          // Restore controls
          controls.style.display = "flex";
          plotEl.style.height = "calc(100vh - 60px)";
        }});
      }}, 100);
    }});
    
    // Add PNG save button functionality
    document.getElementById("save-png").addEventListener("click", function() {{
      const plotEl = document.getElementById("plot");
      const controls = document.getElementById("controls");
      
      // Hide controls and make plot full screen for capture
      controls.style.display = "none";
      plotEl.style.height = "100vh";
      
      // High resolution PNG (2400x2400 at scale 3 = 7200x7200 effective)
      const size = 2400;
      const scale = 3;
      
      setTimeout(function() {{
        Plotly.toImage(plotEl, {{ format: "png", width: size, height: size, scale: scale }}).then(function(dataUrl) {{
          // Download PNG directly
          const link = document.createElement("a");
          link.download = "sphere_view.png";
          link.href = dataUrl;
          link.click();
          
          // Restore controls
          controls.style.display = "flex";
          plotEl.style.height = "calc(100vh - 60px)";
        }});
      }}, 100);
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def make_energy_figure(
    times_null: np.ndarray,
    energy_null: np.ndarray,
    times_mlp: np.ndarray,
    energy_mlp: np.ndarray,
    null_color: str,
    mlp_color: str,
    time_scale: str = "linear",
) -> "plt.Figure":
    """Create energy vs time plot.
    
    Args:
        times_null: Time array for null model
        energy_null: Energy values for null model
        times_mlp: Time array for MLP model
        energy_mlp: Energy values for MLP model
        null_color: Color for null curve
        mlp_color: Color for MLP curve
        time_scale: "linear" or "log"
        
    Note: If null simulation stopped earlier than MLP, the null curve is 
    extended with its final energy value to match MLP's time duration.
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))
    
    # Extend null curve to match MLP time if null stopped earlier
    t_max_mlp = times_mlp[-1] if len(times_mlp) > 0 else 0
    t_max_null = times_null[-1] if len(times_null) > 0 else 0
    
    if t_max_null < t_max_mlp and len(energy_null) > 0:
        # Extend null curve with constant final value
        times_null_ext = np.concatenate([times_null, [t_max_mlp]])
        energy_null_ext = np.concatenate([energy_null, [energy_null[-1]]])
    else:
        times_null_ext = times_null
        energy_null_ext = energy_null
    
    if time_scale == "log":
        # Skip t=0 for log scale
        mask_null = times_null_ext > 0
        mask_mlp = times_mlp > 0
        ax.plot(times_null_ext[mask_null], energy_null_ext[mask_null], color=null_color, linewidth=1.2, label="Null")
        ax.plot(times_mlp[mask_mlp], energy_mlp[mask_mlp], color=mlp_color, linewidth=1.2, label="MLP")
        ax.set_xscale("log")
    else:
        ax.plot(times_null_ext, energy_null_ext, color=null_color, linewidth=1.2, label="Null")
        ax.plot(times_mlp, energy_mlp, color=mlp_color, linewidth=1.2, label="MLP")
    ax.set_xlabel(r"$t$")
    
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=7)
    
    # Use scientific notation for y-axis if values are large
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    
    fig.tight_layout()
    return fig


def compute_c_theta(
    mlp_a: np.ndarray,
    mlp_omega: np.ndarray,
    activation: str,
) -> float:
    """Compute C_θ = Σ_j |ω_j| * (Lip(σ)|a_j|² + (|σ(0)| + Lip(σ)|a_j|)|a_j|).
    
    For relu: Lip(σ) = 1, σ(0) = 0
    For gelu: Lip(σ) ≈ 1.085, σ(0) = 0
    """
    if activation == "relu":
        lip_sigma = 1.0
        sigma_0 = 0.0
    elif activation == "gelu":
        lip_sigma = 1.085  # Approximate Lipschitz constant for GELU
        sigma_0 = 0.0
    else:
        lip_sigma = 1.0
        sigma_0 = 0.0
    
    c_theta = 0.0
    for j in range(mlp_a.shape[0]):
        a_j = mlp_a[j]
        omega_j = mlp_omega[j]
        norm_a_j = np.linalg.norm(a_j)
        norm_omega_j = np.linalg.norm(omega_j)
        
        term = norm_omega_j * (
            lip_sigma * norm_a_j**2 + 
            (sigma_0 + lip_sigma * norm_a_j) * norm_a_j
        )
        c_theta += term
    
    return c_theta


def theoretical_heaviest_mass_bound(beta: float, c_theta: float) -> float:
    """Compute the theoretical bound for heaviest cluster mass.
    
    Formula: (4 + 2*e^(3/2) * C_θ * e^(-β)) / (4 + (3/4)*e^(11/8))
    
    Returns value > 1 if bound is not useful (shouldn't be plotted).
    """
    numerator = 4.0 + 2.0 * np.exp(1.5) * c_theta * np.exp(-beta)
    denominator = 4.0 + 0.75 * np.exp(11.0 / 8.0)
    return numerator / denominator


def make_heaviest_mass_figure(
    betas: np.ndarray,
    heaviest_null: np.ndarray,
    heaviest_mlp: np.ndarray,
    smallest_mlp: Optional[np.ndarray] = None,
    mlp_a: Optional[np.ndarray] = None,
    mlp_omega: Optional[np.ndarray] = None,
    mlp_activation: Optional[str] = None,
    null_color: str = "#ED5E93",
    mlp_color: str = "#0072B2",
    theory_color: str = "#D55E00",
    smallest_color: str = "#E69F00",
) -> plt.Figure:
    """Plot heaviest cluster mass vs sqrt(beta).
    
    Only plots MLP curve and theoretical bound (no null curve).
    If mlp_a, mlp_omega, and mlp_activation are provided, computes and plots theoretical bound.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    sqrt_betas = np.sqrt(betas)
    
    # Plot MLP curve only
    ax.plot(
        sqrt_betas,
        heaviest_mlp,
        "s-",
        color=mlp_color,
        markersize=3,
        linewidth=1.2,
        label="heaviest",
    )
    if smallest_mlp is not None:
        ax.plot(
            sqrt_betas,
            smallest_mlp,
            "o--",
            color=smallest_color,
            markersize=3,
            linewidth=1.0,
            label="smallest",
        )
    
    # Plot theoretical bound if MLP parameters provided
    if mlp_a is not None and mlp_omega is not None and mlp_activation is not None:
        c_theta = compute_c_theta(mlp_a, mlp_omega, mlp_activation)
        theory_values = np.array([theoretical_heaviest_mass_bound(b, c_theta) for b in betas])
        ax.plot(
            sqrt_betas,
            theory_values,
            "^--",
            color=theory_color,
            markersize=3,
            linewidth=1.0,
            label="theory",
        )
    
    # Plot horizontal line at constant 16/(16 + 3*e^(11/8))
    constant_threshold = 16.0 / (16.0 + 3.0 * np.exp(11.0 / 8.0))
    ax.axhline(y=constant_threshold, color="black", linestyle="--", linewidth=1.0)
    
    ax.set_xlabel(r"$\sqrt{\beta}$")
    ax.set_ylabel(r"mass")
    
    # Set y-axis limits based on data (don't start at 0)
    y_values = [np.min(heaviest_mlp), constant_threshold]
    if smallest_mlp is not None and smallest_mlp.size:
        finite_smallest = smallest_mlp[np.isfinite(smallest_mlp)]
        if finite_smallest.size:
            y_values.append(np.min(finite_smallest))
    y_min = min(y_values) - 0.05
    ax.set_ylim(y_min, 1.05)
    
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=7)
    
    fig.tight_layout()
    return fig


def make_all_masses_figure(
    betas: np.ndarray,
    all_masses: list[list[float]],
    mlp_a: Optional[np.ndarray] = None,
    mlp_omega: Optional[np.ndarray] = None,
    mlp_activation: Optional[str] = None,
    mlp_color: str = "#0072B2",
    theory_color: str = "#D55E00",
    point_alpha: float = 0.6,
) -> "plt.Figure":
    """Plot all cluster masses vs sqrt(beta)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sqrt_betas = np.sqrt(betas)
    jitter_width = 0.015
    for x, masses in zip(sqrt_betas, all_masses):
        masses_arr = np.asarray(masses, dtype=float).ravel()
        masses_arr = masses_arr[np.isfinite(masses_arr)]
        if masses_arr.size == 0:
            continue
        if masses_arr.size == 1:
            x_vals = np.array([x])
        else:
            jitter = np.linspace(-jitter_width, jitter_width, masses_arr.size)
            x_vals = x + jitter
        ax.scatter(
            x_vals,
            masses_arr,
            s=10,
            color=mlp_color,
            alpha=point_alpha,
            linewidths=0.0,
        )

    if mlp_a is not None and mlp_omega is not None and mlp_activation is not None:
        c_theta = compute_c_theta(mlp_a, mlp_omega, mlp_activation)
        theory_values = np.array([theoretical_heaviest_mass_bound(b, c_theta) for b in betas])
        ax.plot(sqrt_betas, theory_values, "^--", color=theory_color, markersize=3, linewidth=1.0)

    constant_threshold = 16.0 / (16.0 + 3.0 * np.exp(11.0 / 8.0))
    ax.axhline(y=constant_threshold, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"$\sqrt{\beta}$")
    ax.set_ylabel(r"mass")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=7)

    fig.tight_layout()
    return fig


def make_energy_overlay_figure(
    energy_data: list[dict],
    colors: list[str],
    time_scale: str = "linear",
    show_legend: bool = False,
) -> plt.Figure:
    """Plot energy decay curves for multiple betas in the same figure.
    
    Only MLP curves are plotted (no null/MLP=0 curves) for clarity.
    
    Args:
        energy_data: List of dicts with keys 'beta', 'times_mlp', 'energy_mlp'
        colors: List of colors, one per beta
        time_scale: "linear" or "log"
        show_legend: Whether to show a compact legend with beta values
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, data in enumerate(energy_data):
        color = colors[i % len(colors)]
        beta = data.get("beta", i)
        times_mlp = np.array(data.get("times_mlp", []))
        energy_mlp = np.array(data.get("energy_mlp", []))
        
        # Label only for legend (use compact format)
        label = rf"$\beta={int(beta)}$" if beta == int(beta) else rf"$\beta={beta:.1f}$"
        
        # Only plot MLP curves (solid lines)
        if len(times_mlp) > 0 and len(energy_mlp) > 0:
            if time_scale == "log":
                mask = times_mlp > 0
                ax.plot(times_mlp[mask], energy_mlp[mask], "-", color=color, linewidth=1.2, label=label if show_legend else None)
            else:
                ax.plot(times_mlp, energy_mlp, "-", color=color, linewidth=1.2, label=label if show_legend else None)
    
    if time_scale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(r"$t$")
    
    ax.set_ylabel(r"$\mathsf{E}[\mu_t]$")
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    
    fig.tight_layout()
    return fig
