"""Plot utilities for figure outputs."""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import List, Optional
from numpy.typing import NDArray
import matplotlib as mpl
import numpy as np
from scipy.special import iv

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
(ROOT / ".matplotlib").mkdir(exist_ok=True)

mpl.use("Agg")
_USE_TEX_ENV = os.getenv("WGF_USE_TEX")
if _USE_TEX_ENV is not None:
    _USE_TEX = _USE_TEX_ENV.strip().lower() not in {"0", "false", "no"}
else:
    _USE_TEX = shutil.which("latex") is not None
PLOT_DPI = 1200
BASE_FONT_SIZE = 14
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
TITLE_SIZE = 18
LEGEND_SIZE = 14
mpl.rcParams.update(
    {
        "text.usetex": _USE_TEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "savefig.dpi": PLOT_DPI,
    }
)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_TEX_FALLBACK_APPLIED = False
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Wedge
from scipy.special import erf
from scipy.ndimage import gaussian_filter

from .dynamics import TWO_PI, mlp_drift

POTENTIAL_POS_COLOR = "#009E73"
POTENTIAL_NEG_COLOR =  "#D55E00"
NULL_COLOR ="#ED5E93"
MLP_COLOR =  "#0072B2"
SA_COLOR = "#17BECF"
POINT_COLOR ="#000000"
POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "potential",
    [POTENTIAL_NEG_COLOR, "#FFFFFF", POTENTIAL_POS_COLOR],
)

def _latex_text(label: str) -> str:
    if "$" in label:
        return label
    safe = label.replace("\\", r"\\").replace("_", r"\_").replace("%", r"\%")
    safe = safe.replace(" ", r"\ ")
    return rf"$\mathrm{{{safe}}}$"




def gamma_k_s1(beta: float, k_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute gamma_k for W(q) = beta^-1 * exp(beta q) on S1.

    For d=2, gamma_k = k^2 * W_hat_k, and W_hat_k = I_k(beta) / beta.
    """
    k = np.asarray(k_values, dtype=np.float64)
    if beta == 0.0:
        gamma = np.zeros_like(k)
        gamma[np.isclose(k, 1.0)] = 0.5
        return gamma

    gamma = np.zeros_like(k)
    mask = k != 0.0
    gamma[mask] = (k[mask] * k[mask]) * iv(k[mask], beta) / beta
    return gamma


def plot_gamma(ax, beta: float, k_limit: int, k_max: int) -> None:
    k_int = np.arange(0, k_limit + 1)
    gamma_int = gamma_k_s1(beta, k_int)
    k_display_max = float(k_limit)
    k_cont = np.linspace(0.0, k_display_max, 400)
    gamma_cont = gamma_k_s1(beta, k_cont)

    ax.plot(k_cont, gamma_cont, "--", color=NULL_COLOR, alpha=0.8, linewidth=1.0)
    ax.scatter(k_int, gamma_int, color=NULL_COLOR, s=18)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\gamma_k$")
    ax.set_xlim(0.0, k_display_max)


def _split_wrapped_segment(
    t0: float,
    a0: float,
    t1: float,
    a1: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    k0 = int(np.floor(a0 / TWO_PI))
    k1 = int(np.floor(a1 / TWO_PI))
    a0_mod = a0 % TWO_PI
    a1_mod = a1 % TWO_PI
    if k0 == k1:
        return [((t0, a0_mod), (t1, a1_mod))]

    if a1 > a0:
        boundary = (k0 + 1) * TWO_PI
        frac = (boundary - a0) / (a1 - a0)
        t_mid = t0 + frac * (t1 - t0)
        return [
            ((t0, a0_mod), (t_mid, TWO_PI)),
            ((t_mid, 0.0), (t1, a1_mod)),
        ]

    boundary = k0 * TWO_PI
    frac = (boundary - a0) / (a1 - a0)
    t_mid = t0 + frac * (t1 - t0)
    return [
        ((t0, a0_mod), (t_mid, 0.0)),
        ((t_mid, TWO_PI), (t1, a1_mod)),
    ]


def plot_trajectories(
    ax,
    times: np.ndarray,
    theta_hist: np.ndarray,
    color: str,
    max_particles: int = 250,
    time_stride: int = 1,
    time_scale: str = "linear",
    line_width: float = 0.5,
    line_style: Optional[object] = None,
) -> None:
    """Plot particle trajectories with smooth interpolation and density coloring."""
    if time_stride < 1:
        time_stride = 1
    times_plot = np.asarray(times, dtype=float)
    if time_scale == "log":
        positive = times_plot[times_plot > 0.0]
        if positive.size:
            eps = float(np.min(positive)) * 0.5
        else:
            eps = 1e-6
        times_plot = np.where(times_plot <= 0.0, eps, times_plot)

    times_s = times_plot[::time_stride]
    theta_s = theta_hist[::time_stride]
    n_particles = theta_s.shape[1]
    max_particles = max(1, max_particles)
    step = max(1, n_particles // max_particles)
    particle_idx = np.arange(0, n_particles, step)[:max_particles]
    
    base_color = np.array(to_rgb(color))
    
    # Compute smooth density field for coloring
    density_func = None
    use_density = False
    
    if len(times_s) > 5 and n_particles > 20:
        from scipy.interpolate import RegularGridInterpolator
        
        # Create a smooth density field on a regular grid
        n_time_grid = min(80, len(times_s))
        n_angle_grid = 100
        
        time_grid = np.linspace(times_s[0], times_s[-1], n_time_grid)
        angle_grid = np.linspace(0, TWO_PI, n_angle_grid, endpoint=False)
        
        density_field = np.zeros((n_time_grid, n_angle_grid))
        
        # Compute density using kernel smoothing
        sigma_angle = TWO_PI / 15.0  # Smoothness in angle
        
        for t_idx, t_val in enumerate(time_grid):
            # Find closest time index in original data
            orig_idx = np.argmin(np.abs(times_s - t_val))
            particles_at_t = theta_s[orig_idx, :] % TWO_PI
            
            for a_idx, angle in enumerate(angle_grid):
                # Circular distance
                diff = particles_at_t - angle
                diff = np.arctan2(np.sin(diff), np.cos(diff))
                
                # Gaussian kernel
                weights = np.exp(-0.5 * (diff / sigma_angle) ** 2)
                density_field[t_idx, a_idx] = weights.mean()
        
        # Heavy 2D Gaussian smoothing for truly smooth field
        density_field = gaussian_filter(density_field, sigma=[2.5, 3.0], mode='wrap')
        
        # Create interpolator (regular grid is much faster than RBF)
        density_func = RegularGridInterpolator(
            (time_grid, angle_grid),
            density_field,
            method='cubic',
            bounds_error=False,
            fill_value=None
        )
        
        # Check variation
        density_range = np.max(density_field) - np.min(density_field)
        use_density = density_range > 0.05 * np.max(density_field)
        
        d_min = np.min(density_field)
        d_max = np.max(density_field)
    
    # Interpolate trajectories for smoothness
    if len(times_s) > 2:
        from scipy.interpolate import interp1d
        n_interp = min(len(times_s) * 10, 2000)
        if time_scale == "log":
            times_interp = np.geomspace(times_s[0], times_s[-1], n_interp)
        else:
            times_interp = np.linspace(times_s[0], times_s[-1], n_interp)
    else:
        times_interp = times_s
    
    segments = []
    colors = []
    
    unwrap_discont = 0.95 * np.pi
    min_mix = 0.2
    density_gamma = 0.5

    for idx in particle_idx:
        angles = theta_s[:, idx]
        angles_unwrapped = np.unwrap(angles, discont=unwrap_discont)
        
        # Interpolate
        if len(times_s) > 2:
            interp_func = interp1d(times_s, angles_unwrapped, kind='cubic', 
                                   fill_value='extrapolate', bounds_error=False)
            angles_interp = interp_func(times_interp)
        else:
            angles_interp = angles_unwrapped
        
        t_vals = times_interp
        angles_vals = angles_interp

        for k in range(len(angles_vals) - 1):
            # Get density-based color
            if use_density and density_func is not None:
                t_mid = 0.5 * (t_vals[k] + t_vals[k + 1])
                a_mid = 0.5 * (angles_vals[k] + angles_vals[k + 1])
                a_mid_wrapped = a_mid % TWO_PI
                
                try:
                    # Query interpolated density
                    density = density_func([[t_mid, a_mid_wrapped]])[0]
                    
                    # Normalize
                    if d_max > d_min:
                        norm = (density - d_min) / (d_max - d_min)
                        norm = float(np.clip(norm, 0.0, 1.0))
                    else:
                        norm = 0.5
                    
                    # Gamma for perceptual uniformity
                    norm = norm ** density_gamma
                    mix = min_mix + (1.0 - min_mix) * norm
                    seg_color = base_color * mix + (1.0 - mix)
                    seg_alpha = 0.5 + 0.4 * norm
                except Exception:
                    seg_color = 0.65 * base_color + 0.35
                    seg_alpha = 0.6
            else:
                seg_color = 0.65 * base_color + 0.35
                seg_alpha = 0.6
            
            # Split at 2π boundary
            for seg in _split_wrapped_segment(
                t_vals[k],
                angles_vals[k],
                t_vals[k + 1],
                angles_vals[k + 1],
            ):
                segments.append(seg)
                colors.append((seg_color[0], seg_color[1], seg_color[2], seg_alpha))

    if segments:
        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=line_width,
            antialiaseds=True,
        )
        lc.set_capstyle("round")
        lc.set_joinstyle("round")
        if line_style is not None:
            lc.set_linestyle(line_style)
        ax.add_collection(lc)

    ax.set_ylim(0.0, TWO_PI)
    ax.set_yticks([0.0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\theta$")
    if time_scale == "log":
        ax.set_xscale("log")
    ax.set_xlim(float(times_s[0]), float(times_s[-1]))


def _trajectory_time_bounds(times: np.ndarray, time_scale: str) -> tuple[float, float]:
    times_plot = np.asarray(times, dtype=float)
    if times_plot.size == 0:
        return 0.0, 1.0
    if time_scale == "log":
        positive = times_plot[times_plot > 0.0]
        if positive.size:
            eps = float(np.min(positive)) * 0.5
        else:
            eps = 1e-6
        times_plot = np.where(times_plot <= 0.0, eps, times_plot)
    return float(times_plot[0]), float(times_plot[-1])


def add_trajectory_potential_background(
    ax,
    times: np.ndarray,
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    time_scale: str = "linear",
    alpha: float = 0.3,
    n_theta: int = 720,
) -> None:
    if a.size == 0 or omega.size == 0:
        return
    n_theta = max(2, int(n_theta))
    theta = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
    potential = mlp_potential(theta, a, omega, activation)
    if potential.size == 0:
        return
    max_abs = float(np.max(np.abs(potential)))
    if max_abs <= 0.0:
        return
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    t_min, t_max = _trajectory_time_bounds(times, time_scale)
    ax.imshow(
        potential[:, None],
        extent=(t_min, t_max, 0.0, TWO_PI),
        aspect="auto",
        cmap=POTENTIAL_CMAP,
        norm=norm,
        alpha=alpha,
        origin="lower",
        interpolation="nearest",
        zorder=0,
    )


def _boundary_angles(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.array([0.0, TWO_PI])
    angles = []
    for vec in a:
        phi = np.arctan2(vec[1], vec[0]) + 0.5 * np.pi
        angles.append(phi % TWO_PI)
        angles.append((phi + np.pi) % TWO_PI)
    unique = np.unique(np.asarray(angles))
    unique.sort()
    return unique


def _region_gray(a: np.ndarray, angle: float) -> float:
    if a.size == 0:
        return 1.0
    x = np.array([np.cos(angle), np.sin(angle)])
    signs = (a @ x) < 0.0
    count = int(signs.sum())
    return 1.0 - count / a.shape[0]


def _add_region_shading(ax, a: np.ndarray, radius: float = 1.0, alpha: float = 0.35) -> None:
    boundaries = _boundary_angles(a)
    if boundaries.size == 2:
        sector_angles = [(0.0, TWO_PI)]
    else:
        extended = np.concatenate([boundaries, boundaries[:1] + TWO_PI])
        sector_angles = [(extended[i], extended[i + 1]) for i in range(len(boundaries))]

    for start, end in sector_angles:
        mid = 0.5 * (start + end)
        gray = _region_gray(a, mid)
        patch = Wedge(
            (0.0, 0.0),
            radius,
            np.degrees(start),
            np.degrees(end),
            facecolor=(gray, gray, gray),
            edgecolor="none",
            alpha=alpha,
            zorder=0,
        )
        ax.add_patch(patch)


def plot_circular_histogram(
    ax,
    theta: np.ndarray,
    a: np.ndarray,
    bins: int = 120,
    max_bar: float = 0.35,
    color: str = MLP_COLOR,
    shade_regions: bool = True,
    bar_alpha: float = 0.85,
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    if shade_regions:
        _add_region_shading(ax, a)
    circle = plt.Circle((0.0, 0.0), 1.0, edgecolor="black", facecolor="none", linewidth=1.0, zorder=3)
    ax.add_patch(circle)

    counts, edges = np.histogram(theta, bins=bins, range=(0.0, TWO_PI), density=False)
    max_count = max(1, counts.max())

    for count, start, end in zip(counts, edges[:-1], edges[1:]):
        radius = 1.0 + max_bar * (count / max_count)
        patch = Wedge(
            (0.0, 0.0),
            radius,
            np.degrees(start),
            np.degrees(end),
            width=radius - 1.0,
            facecolor=color,
            edgecolor="none",
            alpha=bar_alpha,
            zorder=2,
        )
        ax.add_patch(patch)


def plot_overlaid_circular_histograms(
    ax,
    thetas: list[np.ndarray],
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    bins: int = 120,
    max_bar: float = 0.35,
    color: str = MLP_COLOR,
    bar_alpha: float = 0.35,
    shade_regions: bool = True,
    show_potential: bool = False,
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    if shade_regions:
        _add_region_shading(ax, a)
    circle = plt.Circle((0.0, 0.0), 1.0, edgecolor="black", facecolor="none", linewidth=1.0, zorder=3)
    ax.add_patch(circle)

    if show_potential and a.size:
        potential_bins = max(360, bins * 4)
        theta_grid = np.linspace(0.0, TWO_PI, potential_bins, endpoint=False)
        potential = mlp_potential(theta_grid, a, omega, activation)
        _draw_potential_inside(ax, theta_grid, potential)

    edges = np.linspace(0.0, TWO_PI, bins + 1)
    counts_list = [np.histogram(theta, bins=edges, density=False)[0] for theta in thetas]
    max_count = 1
    for counts in counts_list:
        if counts.size:
            max_count = max(max_count, int(counts.max()))

    for counts in counts_list:
        for count, start, end in zip(counts, edges[:-1], edges[1:]):
            radius = 1.0 + max_bar * (count / max_count)
            patch = Wedge(
                (0.0, 0.0),
                radius,
                np.degrees(start),
                np.degrees(end),
                width=radius - 1.0,
                facecolor=color,
                edgecolor="none",
                alpha=bar_alpha,
                zorder=2,
            )
            ax.add_patch(patch)


def plot_histogram_with_potential(
    ax,
    theta: np.ndarray,
    null_theta: Optional[np.ndarray],
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    bins: int = 80,
    show_potential: bool = True,
    mlp_color: str = MLP_COLOR,
) -> None:
    counts, edges = np.histogram(theta, bins=bins, range=(0.0, TWO_PI), density=False)
    width = edges[1] - edges[0]
    max_height = counts.max() if counts.size else 1.0

    counts_null = None
    if null_theta is not None and null_theta.size:
        counts_null, _ = np.histogram(null_theta, bins=edges, density=False)
        max_height = max(max_height, counts_null.max())

    if show_potential and a.size:
        grid = np.linspace(0.0, TWO_PI, bins * 4, endpoint=False)
        potential = mlp_potential(grid, a, omega, activation)
        max_abs = np.max(np.abs(potential))
        if max_abs > 0.0:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
            ax.imshow(
                potential[None, :],
                extent=(0.0, TWO_PI, 0.0, max_height * 1.05),
                aspect="auto",
                cmap=POTENTIAL_CMAP,
                alpha=0.35,
                norm=norm,
                origin="lower",
            )

    ax.bar(
        edges[:-1],
        counts,
        width=width,
        align="edge",
        color=mlp_color,
        alpha=1.0,
        edgecolor="none",
    )

    if counts_null is not None:
        ax.bar(
            edges[:-1],
            counts_null,
            width=width * 0.8,
            align="edge",
            color=NULL_COLOR,
            alpha=1.0,
            edgecolor="none",
        )
    ax.set_xlim(0.0, TWO_PI)
    ax.set_xticks([0.0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\mathrm{count}$")


def make_trajectory_figure(
    times: np.ndarray,
    theta_hist: np.ndarray,
    color: str,
    time_scale: str = "linear",
    a: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    activation: str = "relu",
    show_potential: bool = False,
    potential_alpha: float = 0.3,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    if show_potential and a is not None and omega is not None:
        add_trajectory_potential_background(
            ax,
            times,
            a,
            omega,
            activation,
            time_scale=time_scale,
            alpha=potential_alpha,
        )
    plot_trajectories(
        ax,
        times,
        theta_hist,
        color=color,
        time_scale=time_scale,
        line_style="solid",
    )
    return fig


def make_histogram_figure(
    theta: np.ndarray,
    null_theta: Optional[np.ndarray],
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    include_null: bool = True,
    show_potential: bool = True,
    mlp_color: str = MLP_COLOR,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    plot_histogram_with_potential(
        ax,
        theta,
        null_theta if include_null else None,
        a=a,
        omega=omega,
        activation=activation,
        show_potential=show_potential,
        mlp_color=mlp_color,
    )
    return fig


def make_mlp_potential_figure(
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    n_points: int = 720,
    line_color: str = MLP_COLOR,
    line_width: float = 1.6,
) -> plt.Figure:
    theta = np.linspace(0.0, TWO_PI, n_points, endpoint=True)
    potential = mlp_potential(theta, a, omega, activation)
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    ax.plot(theta, potential, color=line_color, linewidth=line_width)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlim(0.0, TWO_PI)
    ax.set_xticks([0.0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$v(\theta)$")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    return fig


def make_gamma_figure(beta: float, k_limit: int, k_max: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    plot_gamma(ax, beta, k_limit, k_max)
    return fig


def save_figure(
    fig,
    output_stem: Path,
    formats: tuple[str, ...] = ("pdf",),
    dpi_by_format: Optional[dict[str, int]] = None,
    dpi: Optional[int] = None,
) -> None:
    global _TEX_FALLBACK_APPLIED
    for fmt in formats:
        target_dpi = PLOT_DPI
        if dpi is not None:
            target_dpi = int(dpi)
        if dpi_by_format and fmt in dpi_by_format:
            target_dpi = int(dpi_by_format[fmt])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="constrained_layout not applied*",
                category=UserWarning,
            )
            try:
                fig.savefig(output_stem.with_suffix(f".{fmt}"), dpi=target_dpi)
            except RuntimeError as exc:
                message = str(exc).lower()
                tex_error = "latex was not able to process" in message or "type1cm.sty" in message
                if not tex_error or _TEX_FALLBACK_APPLIED:
                    raise
                _TEX_FALLBACK_APPLIED = True
                mpl.rcParams.update({"text.usetex": False})
                print("Warning: LaTeX unavailable; falling back to mathtext for plots.")
                fig.savefig(output_stem.with_suffix(f".{fmt}"), dpi=target_dpi)


def make_mlp_scale_stop_time_figure(
    scales: np.ndarray,
    stop_times: np.ndarray,
    line_color: str = MLP_COLOR,
    point_size: float = 3.0,
    line_width: float = 1.2,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(scales, stop_times, color=line_color, linewidth=line_width)
    ax.scatter(scales, stop_times, color=line_color, s=point_size, zorder=3)
    ax.set_xlabel(r"$|\omega_j|$")
    ax.set_ylabel("Final time")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    return fig


def make_mlp_scale_stop_time_comparison_figure(
    usa_scales: np.ndarray,
    usa_stop_times: np.ndarray,
    sa_scales: np.ndarray,
    sa_stop_times: np.ndarray,
    point_size: float = 3.0,
    line_width: float = 1.5,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(usa_scales, usa_stop_times, color=MLP_COLOR, linewidth=line_width)
    ax.plot(sa_scales, sa_stop_times, color=SA_COLOR, linewidth=line_width, linestyle="--")
    ax.scatter(usa_scales, usa_stop_times, color=MLP_COLOR, s=point_size, zorder=3)
    ax.scatter(sa_scales, sa_stop_times, color=SA_COLOR, s=point_size, zorder=3)
    ax.set_xlabel(r"$|\omega_j|$")
    ax.set_ylabel("Convergence time (s)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    return fig



def make_cluster_bar_plot(
    sqrt_betas: np.ndarray,
    mlp_mean: np.ndarray,
    ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    if sqrt_betas.size == 1:
        # Single beta: use bar with fixed width
        width = 0.5
        ax.bar(sqrt_betas, mlp_mean, width=width, color=MLP_COLOR, edgecolor="none", alpha=0.9)
        ax.set_xlim(float(sqrt_betas[0]) - 1.0, float(sqrt_betas[0]) + 1.0)
    else:
        diffs = np.diff(np.sort(sqrt_betas))
        width = 0.9 * float(np.min(diffs)) if diffs.size else 0.7
        ax.bar(
            sqrt_betas,
            mlp_mean,
            width=width,
            color=MLP_COLOR,
            edgecolor="none",
            alpha=0.9,
        )
    ax.set_xlabel(r"$\sqrt{\beta}$")
    ax.set_ylabel(_latex_text(ylabel))
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def make_cluster_bar_plot_with_null(
    sqrt_betas: np.ndarray,
    null_mean: np.ndarray,
    mlp_mean: np.ndarray,
    ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    if sqrt_betas.size == 1:
        # Single beta: use bars with fixed width
        width = 0.3
        ax.bar(sqrt_betas - width / 2.0, null_mean, width=width, color=NULL_COLOR, alpha=0.9)
        ax.bar(sqrt_betas + width / 2.0, mlp_mean, width=width, color=MLP_COLOR, alpha=0.9)
        ax.set_xlim(float(sqrt_betas[0]) - 1.0, float(sqrt_betas[0]) + 1.0)
    else:
        order = np.argsort(sqrt_betas)
        xs = sqrt_betas[order]
        null_vals = null_mean[order]
        mlp_vals = mlp_mean[order]
        diffs = np.diff(xs)
        base_width = 0.7 if diffs.size == 0 else 0.9 * float(np.min(diffs))
        width = base_width / 2.0
        ax.bar(xs - width / 2.0, null_vals, width=width, color=NULL_COLOR, alpha=0.9)
        ax.bar(xs + width / 2.0, mlp_vals, width=width, color=MLP_COLOR, alpha=0.9)
    ax.set_xlabel(r"$\sqrt{\beta}$")
    ax.set_ylabel(_latex_text(ylabel))
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_histogram_1d(
    ax,
    theta: np.ndarray,
    color: str,
    bins: int = 80,
) -> None:
    counts, edges = np.histogram(theta, bins=bins, range=(0.0, TWO_PI), density=False)
    width = edges[1] - edges[0]
    ax.bar(
        edges[:-1],
        counts,
        width=width,
        align="edge",
        color=color,
        alpha=1.0,
        edgecolor="none",
    )
    ax.set_xlim(0.0, TWO_PI)
    ax.set_xticks([0.0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\mathrm{count}$")


def _primitive_relu(t: np.ndarray) -> np.ndarray:
    return np.maximum(t, 0.0) ** 2


def _primitive_gelu(t: np.ndarray) -> np.ndarray:
    erf_term = erf(t / np.sqrt(2.0))
    return 2.0 * (
        0.5 * t * t
        + (0.5 * t * t - 0.5) * erf_term
        + t * np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)
    )


def mlp_potential(theta: np.ndarray, a: np.ndarray, omega: np.ndarray, activation: str) -> np.ndarray:
    if a.size == 0:
        return np.zeros_like(theta)
    x = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    z = x @ a.T
    if activation == "relu":
        phi = _primitive_relu(z)
    elif activation == "gelu":
        phi = _primitive_gelu(z)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    omega_scalar = (omega * a).sum(axis=1)
    return phi @ omega_scalar


def _draw_potential_inside(
    ax,
    theta: np.ndarray,
    potential: np.ndarray,
    radius: float = 1.0,
    depth_scale: float = 1.0,
    alpha: float = 0.35,
) -> None:
    if potential.size == 0:
        return
    if theta.size == 0:
        return
    if np.allclose(potential, 0.0):
        return
    step = TWO_PI / float(theta.size)
    abs_potential = np.abs(potential)
    max_abs = float(np.max(abs_potential))
    if max_abs <= 0.0:
        return
    scale = max_abs / max(depth_scale, 1e-12)
    depths = radius * (1.0 - np.exp(-abs_potential / scale))
    for ang, depth, value in zip(theta, depths, potential):
        inner = max(0.0, radius - depth)
        if inner >= radius:
            continue
        color = POTENTIAL_POS_COLOR if value >= 0.0 else POTENTIAL_NEG_COLOR
        patch = Wedge(
            (0.0, 0.0),
            radius,
            np.degrees(ang),
            np.degrees(ang + step),
            width=radius - inner,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)


def make_mlp_figure(
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    title: str,
    gradient_mlp: bool,
    num_points: int = 256,
    vector_points: int = 32,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    _add_region_shading(ax, a)
    circle = plt.Circle((0.0, 0.0), 1.0, edgecolor="black", facecolor="none", linewidth=1.0)
    ax.add_patch(circle)

    for vec in a:
        normal = np.array([-vec[1], vec[0]])
        ax.plot([-normal[0], normal[0]], [-normal[1], normal[1]], color="gray", alpha=0.5, linewidth=1.0)

    theta = np.linspace(0.0, TWO_PI, num_points, endpoint=False)

    if gradient_mlp and a.size:
        potential = mlp_potential(theta, a, omega, activation)
        centered = potential - potential.mean()
        scale = 0.25 / (np.max(np.abs(centered)) + 1e-8)
        radius = 1.0 + scale * centered
        coords = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        segments = np.stack([coords[:-1], coords[1:]], axis=1)
        segments = np.concatenate([segments, coords[-1:, None, :].repeat(2, axis=1)], axis=0)
        segments[-1, 1, :] = coords[0]
        colors = np.concatenate([potential[:-1], potential[-1:]])
        max_abs = np.max(np.abs(potential))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs) if max_abs > 0.0 else None
        line = LineCollection(segments, array=colors, cmap=POTENTIAL_CMAP, norm=norm, linewidth=3.0)
        ax.add_collection(line)

    class _Params:
        def __init__(self, a: np.ndarray, omega: np.ndarray, activation: str) -> None:
            self.a = a
            self.omega = omega
            self.activation = activation

    field_theta = np.linspace(0.0, TWO_PI, vector_points, endpoint=False)
    drift = mlp_drift(field_theta, _Params(a, omega, activation))
    tangent = np.stack([-np.sin(field_theta), np.cos(field_theta)], axis=1)
    field = drift[:, None] * tangent
    ax.quiver(
        np.cos(field_theta),
        np.sin(field_theta),
        field[:, 0],
        field[:, 1],
        angles="xy",
        scale_units="xy",
        scale=4.0,
        color=MLP_COLOR,
        width=0.004,
    )

    ax.set_title(title)
    return fig


def make_histogram_frame(
    theta: np.ndarray,
    time_value: float,
    beta: float,
    n_particles: int,
    mlp_title: str,
    attention_label: str,
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    color: str,
    shade_regions: bool = True,
    bins: int = 120,
    max_bar: float = 0.35,
    show_potential: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_circular_histogram(
        ax,
        theta,
        a,
        bins=bins,
        max_bar=max_bar,
        color=color,
        shade_regions=shade_regions,
    )

    if show_potential and a.size:
        potential_bins = max(1440, bins * 12)
        theta_grid = np.linspace(0.0, TWO_PI, potential_bins, endpoint=False)
        potential = mlp_potential(theta_grid, a, omega, activation)
        _draw_potential_inside(ax, theta_grid, potential)

    ax.scatter(np.cos(theta), np.sin(theta), s=12, color=color, alpha=0.6, zorder=4)
    ax.set_title(
        rf"$\mathrm{{{attention_label}}},\ t={time_value:.3f},\ \beta={beta:g},\ "
        rf"N={n_particles},\ {mlp_title}$"
    )
    fig.subplots_adjust(top=0.88)

    return fig


def make_histogram_comparison_frame(
    theta_null: np.ndarray,
    theta_mlp: np.ndarray,
    time_value: float,
    beta: float,
    n_particles: int,
    mlp_title: str,
    attention_label: str,
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    bins: int = 120,
    max_bar: float = 0.35,
    show_potential: bool = True,
    mlp_color: str = MLP_COLOR,
) -> plt.Figure:
    fig, axes = plt.subplots(ncols=2, figsize=(11.0, 5.5))

    a_null = np.empty((0, 2))
    omega_null = np.empty((0, 2))
    plot_circular_histogram(axes[0], theta_null, a_null, bins=bins, max_bar=max_bar, color=NULL_COLOR)
    plot_circular_histogram(axes[1], theta_mlp, a, bins=bins, max_bar=max_bar, color=mlp_color)

    if show_potential and a.size:
        potential_bins = max(1440, bins * 12)
        theta_grid = np.linspace(0.0, TWO_PI, potential_bins, endpoint=False)
        potential = mlp_potential(theta_grid, a, omega, activation)
        _draw_potential_inside(axes[1], theta_grid, potential)

    axes[0].scatter(
        np.cos(theta_null),
        np.sin(theta_null),
        s=12,
        color=NULL_COLOR,
        alpha=0.6,
        zorder=4,
    )
    axes[1].scatter(
        np.cos(theta_mlp),
        np.sin(theta_mlp),
        s=12,
        color=mlp_color,
        alpha=0.6,
        zorder=4,
    )

    axes[0].set_title(r"$\mathrm{MLP}\,=\,0$")
    axes[1].set_title(rf"$\mathrm{{std(MLP)}}\,=\,{mlp_title}$")
    fig.suptitle(
        rf"$\mathrm{{{attention_label}}},\ t={time_value:.3f},\ \beta={beta:g},\ "
        rf"N={n_particles}$"
    )
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.86, wspace=0.2)
    return fig


def make_total_clusters_figure(
    thetas: list[np.ndarray],
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    color: str,
    bins: int = 120,
    max_bar: float = 0.35,
    bar_alpha: float = 0.35,
    shade_regions: bool = True,
    show_potential: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_overlaid_circular_histograms(
        ax,
        thetas,
        a,
        omega,
        activation,
        bins=bins,
        max_bar=max_bar,
        color=color,
        bar_alpha=bar_alpha,
        shade_regions=shade_regions,
        show_potential=show_potential,
    )
    return fig


def make_field_frame(
    theta: np.ndarray,
    field: np.ndarray,
    attention_field: Optional[np.ndarray],
    mlp_field: Optional[np.ndarray],
    potential: Optional[np.ndarray],
    time_value: float,
    beta: float,
    n_particles: int,
    mlp_title: str,
    attention_label: str,
    y_limits: tuple[float, float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y_min, y_max = y_limits

    if potential is not None and potential.size:
        max_abs = np.max(np.abs(potential))
        if max_abs > 0.0:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
            ax.imshow(
                potential[None, :],
                extent=(0.0, TWO_PI, y_min, y_max),
                aspect="auto",
                cmap=POTENTIAL_CMAP,
                alpha=0.35,
                norm=norm,
                origin="lower",
            )

    ax.plot(theta, field, color="black", linewidth=1.6, label=r"$u$")
    if attention_field is not None:
        ax.plot(
            theta,
            attention_field,
            color=NULL_COLOR,
            linewidth=1.2,
            linestyle="--",
            label=r"$u_{\mathrm{att}}$",
        )
    if mlp_field is not None:
        ax.plot(
            theta,
            mlp_field,
            color=MLP_COLOR,
            linewidth=1.2,
            linestyle="--",
            label=r"$u_{\mathrm{mlp}}$",
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0.0, TWO_PI)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0.0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$u(\theta)$")
    ax.set_title(
        rf"$\mathrm{{{attention_label}}},\ t={time_value:.3f},\ \beta={beta:g},\ "
        rf"N={n_particles},\ {mlp_title}$"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    return fig


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
    null_color: str = NULL_COLOR,
    mlp_color: str = MLP_COLOR,
    theory_color: str = "#d62728",
    smallest_color: str = MLP_COLOR,
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
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.4,
        linewidth=2.0,
        label="heaviest",
        zorder=3,
    )
    if smallest_mlp is not None:
        ax.plot(
            sqrt_betas,
            smallest_mlp,
            "o--",
            color=smallest_color,
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            linewidth=1.6,
            label="smallest",
            zorder=2,
        )
    
    # Plot theoretical bound if MLP parameters provided
    if mlp_a is not None and mlp_omega is not None and mlp_activation is not None:
        c_theta = compute_c_theta(mlp_a, mlp_omega, mlp_activation)
        theory_values = np.array([theoretical_heaviest_mass_bound(b, c_theta) for b in betas])
        ax.plot(
            sqrt_betas,
            theory_values,
            linestyle="--",
            marker="D",
            color=theory_color,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.4,
            linewidth=1.6,
            label="theory",
            zorder=4,
        )
    
    # Plot horizontal line at constant 16/(16 + 3*e^(11/8))
    constant_threshold = 16.0 / (16.0 + 3.0 * np.exp(11.0 / 8.0))
    ax.axhline(y=constant_threshold, color="black", linestyle="--", linewidth=0.8, zorder=1)
    
    # Override the global large label size for this compact plot.
    ax.set_xlabel(r"$\sqrt{\beta}$", fontsize=9)
    ax.set_ylabel("")
    
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
    mlp_color: str = MLP_COLOR,
    theory_color: str = "#d62728",
    point_alpha: float = 1.0,
) -> plt.Figure:
    """Plot all cluster masses vs sqrt(beta)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    sqrt_betas = np.sqrt(betas)
    masses_by_beta = []
    max_count = 0
    for masses in all_masses:
        masses_arr = np.asarray(masses, dtype=float).ravel()
        masses_arr = masses_arr[np.isfinite(masses_arr)]
        if masses_arr.size:
            masses_arr = np.sort(masses_arr)[::-1]
        masses_by_beta.append(masses_arr)
        max_count = max(max_count, masses_arr.size)

    if max_count:
        mass_matrix = np.full((len(betas), max_count), np.nan, dtype=float)
        for i, masses_arr in enumerate(masses_by_beta):
            if masses_arr.size:
                mass_matrix[i, : masses_arr.size] = masses_arr
        for j in range(max_count):
            y_vals = mass_matrix[:, j]
            mask = np.isfinite(y_vals)
            if np.any(mask):
                line_width = 1.8 if j == 0 else 0.8
                marker = "o" if j == 0 else "^"
                marker_size = 6 if j == 0 else 5
                ax.plot(
                    sqrt_betas[mask],
                    y_vals[mask],
                    marker=marker,
                    linestyle="-",
                    color=mlp_color,
                    markersize=marker_size,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    linewidth=line_width,
                    alpha=point_alpha,
                    zorder=2,
                )

    if mlp_a is not None and mlp_omega is not None and mlp_activation is not None:
        c_theta = compute_c_theta(mlp_a, mlp_omega, mlp_activation)
        theory_values = np.array([theoretical_heaviest_mass_bound(b, c_theta) for b in betas])
        ax.plot(
            sqrt_betas,
            theory_values,
            linestyle="--",
            color=theory_color,
            linewidth=1.8,
            zorder=4,
        )

    constant_threshold = 16.0 / (16.0 + 3.0 * np.exp(11.0 / 8.0))
    ax.axhline(y=constant_threshold, color="black", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_xlabel(r"$\sqrt{\beta}$", fontsize=9)
    ax.set_ylabel("")

    y_values = [constant_threshold]
    for masses_arr in masses_by_beta:
        if masses_arr.size:
            y_values.append(np.min(masses_arr))
    y_min = min(y_values) - 0.05 if y_values else 0.0
    ax.set_ylim(y_min, 1.05)
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
        ax.set_xlabel(r"$t$ (log)")
    else:
        ax.set_xlabel(r"$t$")
    
    ax.set_ylabel(r"$\mathsf{E}[\mu_t]$")
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    
    fig.tight_layout()
    return fig
