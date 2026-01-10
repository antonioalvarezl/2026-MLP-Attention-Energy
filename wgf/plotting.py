"""Plot utilities for Figure-2-style outputs."""
from __future__ import annotations

import os
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
PLOT_DPI = 1200
BASE_FONT_SIZE = 14
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
TITLE_SIZE = 18
LEGEND_SIZE = 14
mpl.rcParams.update(
    {
        "text.usetex": True,
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.patches import Wedge
from scipy.special import erf

from .dynamics import TWO_PI, mlp_drift

POTENTIAL_POS_COLOR = "#009E73"
POTENTIAL_NEG_COLOR =  "#D55E00"
NULL_COLOR ="#ED5E93"
MLP_COLOR =  "#0072B2"
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
    return (k * k) * iv(k, beta) / beta


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


def _density_grid(
    times: np.ndarray,
    theta_hist: np.ndarray,
    angle_bins: int,
    time_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_bins = min(time_bins, len(times))
    time_edges = np.linspace(times[0], times[-1], time_bins + 1)
    angle_edges = np.linspace(0.0, TWO_PI, angle_bins + 1)

    t_grid = np.repeat(times, theta_hist.shape[1])
    a_grid = theta_hist.reshape(-1)
    hist, _, _ = np.histogram2d(t_grid, a_grid, bins=[time_edges, angle_edges])
    return time_edges, angle_edges, hist


def _density_by_bin(
    times: np.ndarray,
    theta_hist: np.ndarray,
    angle_bins: int,
    time_bins: int,
) -> np.ndarray:
    time_edges, angle_edges, hist = _density_grid(times, theta_hist, angle_bins, time_bins)

    t_grid = np.repeat(times, theta_hist.shape[1])
    a_grid = theta_hist.reshape(-1)
    t_idx = np.searchsorted(time_edges, t_grid, side="right") - 1
    a_idx = np.searchsorted(angle_edges, a_grid, side="right") - 1
    t_idx = np.clip(t_idx, 0, hist.shape[0] - 1)
    a_idx = np.clip(a_idx, 0, hist.shape[1] - 1)

    return hist[t_idx, a_idx]


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


def _circular_mean(angles: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return 0.0
    mean = np.arctan2(sin_mean, cos_mean)
    if mean < 0.0:
        mean += TWO_PI
    return mean


def _align_unwrapped(angles_unwrapped: np.ndarray, reference: float) -> np.ndarray:
    if angles_unwrapped.size == 0:
        return angles_unwrapped
    shift = TWO_PI * np.round((reference - angles_unwrapped[-1]) / TWO_PI)
    return angles_unwrapped + shift


def plot_trajectories(
    ax,
    times: np.ndarray,
    theta_hist: np.ndarray,
    color: str,
    angle_bins: int = 120,
    time_bins: int = 200,
    max_particles: int = 200,
    time_stride: int = 5,
    time_scale: str = "linear",
    line_width: float = 0.6,
    line_style: Optional[object] = None,
) -> None:
    # Subsample trajectories but skip density weighting.
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
    ref_angle = _circular_mean(theta_hist[-1])

    base = np.array(to_rgb(color))
    line_color = 0.7 * base + 0.3
    line_alpha = 1.0
    segments = []

    for idx in particle_idx:
        angles = theta_s[:, idx]
        angles_unwrapped = np.unwrap(angles, discont=np.pi)
        angles_unwrapped = _align_unwrapped(angles_unwrapped, ref_angle)
        t_vals = times_s

        for k in range(len(angles) - 1):
            for seg in _split_wrapped_segment(
                t_vals[k],
                angles_unwrapped[k],
                t_vals[k + 1],
                angles_unwrapped[k + 1],
            ):
                segments.append(seg)

    if segments:
        lc = LineCollection(
            segments,
            colors=[(line_color[0], line_color[1], line_color[2], line_alpha)],
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
) -> None:
    counts, edges = np.histogram(theta, bins=bins, range=(0.0, TWO_PI), density=True)
    width = edges[1] - edges[0]
    max_height = counts.max() if counts.size else 1.0

    counts_null = None
    if null_theta is not None and null_theta.size:
        counts_null, _ = np.histogram(null_theta, bins=edges, density=True)
        max_height = max(max_height, counts_null.max())

    if a.size:
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
        color=MLP_COLOR,
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
    ax.set_ylabel(r"$\rho(\theta)$")


def make_figure_mlp(
    beta: float,
    k_max: int,
    null_times: List[np.ndarray] | np.ndarray,
    null_histories: List[np.ndarray],
    mlp_times: List[np.ndarray] | np.ndarray,
    mlp_histories: List[np.ndarray],
    a: np.ndarray,
    omega: np.ndarray,
    activation: str,
    row_labels: List[str],
    time_scale: str = "linear",
    null_title: Optional[str] = None,
    mlp_title: Optional[str] = None,
):
    n_rows = len(mlp_histories)
    if isinstance(null_times, np.ndarray):
        null_times_list = [null_times] * n_rows
    else:
        null_times_list = null_times
    if isinstance(mlp_times, np.ndarray):
        mlp_times_list = [mlp_times] * n_rows
    else:
        mlp_times_list = mlp_times

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=3,
        figsize=(12.5, 4.0 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row, (null_hist, mlp_hist, label) in enumerate(zip(null_histories, mlp_histories, row_labels)):
        plot_trajectories(
            axes[row, 0],
            null_times_list[row],
            null_hist,
            color=NULL_COLOR,
            time_scale=time_scale,
            line_style="solid",
        )
        plot_trajectories(
            axes[row, 1],
            mlp_times_list[row],
            mlp_hist,
            color=MLP_COLOR,
            time_scale=time_scale,
            line_style="solid",
        )
        plot_histogram_with_potential(
            axes[row, 2],
            mlp_hist[-1],
            null_hist[-1],
            a=a,
            omega=omega,
            activation=activation,
        )

    return fig


def save_figure(fig, output_stem: Path, formats: tuple[str, ...] = ("pdf",)) -> None:
    for fmt in formats:
        fig.savefig(output_stem.with_suffix(f".{fmt}"), dpi=PLOT_DPI)


def make_cluster_stats_plot(
    sqrt_betas: np.ndarray,
    null_mean: np.ndarray,
    null_std: np.ndarray,
    mlp_mean: np.ndarray,
    mlp_std: np.ndarray,
    ylabel: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.errorbar(sqrt_betas, null_mean, yerr=null_std, fmt="o-", label=r"MLP$_{\mathrm{null}}$")
    ax.errorbar(sqrt_betas, mlp_mean, yerr=mlp_std, fmt="o-", label=r"$\mathrm{MLP}$")
    ax.set_xlabel(r"$\sqrt{\beta}$")
    ax.set_ylabel(_latex_text(ylabel))
    ax.set_title(_latex_text(title))
    ax.grid(True, alpha=0.3)
    return fig


def make_cluster_bar_plot(
    sqrt_betas: np.ndarray,
    mlp_mean: np.ndarray,
    ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    if sqrt_betas.size == 1:
        ax.scatter(sqrt_betas, mlp_mean, color=MLP_COLOR, s=60)
        ax.set_xlim(float(sqrt_betas[0]) - 0.5, float(sqrt_betas[0]) + 0.5)
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
        ax.scatter(sqrt_betas, null_mean, color=NULL_COLOR, s=60)
        ax.scatter(sqrt_betas, mlp_mean, color=MLP_COLOR, s=60)
        ax.set_xlim(float(sqrt_betas[0]) - 0.5, float(sqrt_betas[0]) + 0.5)
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
    counts, edges = np.histogram(theta, bins=bins, range=(0.0, TWO_PI), density=True)
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
    ax.set_ylabel(r"$\rho(\theta)$")


def make_convergence_figure(
    betas: List[float],
    times_list: List[np.ndarray],
    histories: List[np.ndarray],
    k_max_list: List[int],
    k_limit: int,
    color: str,
    time_scale: str = "linear",
) -> plt.Figure:
    n_rows = len(betas)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=3,
        figsize=(12.5, 4.0 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row, (beta, times, theta_hist, k_max) in enumerate(zip(betas, times_list, histories, k_max_list)):
        plot_gamma(axes[row, 0], beta, k_limit, k_max)
        plot_trajectories(axes[row, 1], times, theta_hist, color=color, time_scale=time_scale)
        plot_histogram_1d(axes[row, 2], theta_hist[-1], color=color)

    return fig


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
    if not gradient_mlp:
        raise ValueError("MLP plot requires a gradient field (gradient_MLP=True).")
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

    if gradient_mlp:
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
    bins: int = 120,
    max_bar: float = 0.35,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_circular_histogram(ax, theta, a, bins=bins, max_bar=max_bar, color=color)

    potential_bins = max(360, bins * 4)
    theta_grid = np.linspace(0.0, TWO_PI, potential_bins, endpoint=False)
    potential = mlp_potential(theta_grid, a, omega, activation)
    _draw_potential_inside(ax, theta_grid, potential)

    ax.scatter(np.cos(theta), np.sin(theta), s=6, color=POINT_COLOR, alpha=0.6, zorder=4)
    ax.set_title(
        rf"$\mathrm{{{attention_label}}},\ t={time_value:.3f},\ \beta={beta:g},\ "
        rf"N={n_particles},\ {mlp_title}$"
    )

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
) -> plt.Figure:
    fig, axes = plt.subplots(ncols=2, figsize=(11.0, 5.5), constrained_layout=True)

    a_null = np.empty((0, 2))
    omega_null = np.empty((0, 2))
    plot_circular_histogram(axes[0], theta_null, a_null, bins=bins, max_bar=max_bar, color=NULL_COLOR)
    plot_circular_histogram(axes[1], theta_mlp, a, bins=bins, max_bar=max_bar, color=MLP_COLOR)

    potential_bins = max(360, bins * 4)
    theta_grid = np.linspace(0.0, TWO_PI, potential_bins, endpoint=False)
    potential = mlp_potential(theta_grid, a, omega, activation)
    _draw_potential_inside(axes[1], theta_grid, potential)

    axes[0].scatter(
        np.cos(theta_null),
        np.sin(theta_null),
        s=6,
        color=POINT_COLOR,
        alpha=0.6,
        zorder=4,
    )
    axes[1].scatter(
        np.cos(theta_mlp),
        np.sin(theta_mlp),
        s=6,
        color=POINT_COLOR,
        alpha=0.6,
        zorder=4,
    )

    axes[0].set_title(r"$\mathrm{MLP}\,=\,0$")
    axes[1].set_title(rf"$\mathrm{{std(MLP)}}\,=\,{mlp_title}$")
    fig.suptitle(
        rf"$\mathrm{{{attention_label}}},\ t={time_value:.3f},\ \beta={beta:g},\ "
        rf"N={n_particles}$"
    )
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
    potential: np.ndarray,
    time_value: float,
    beta: float,
    n_particles: int,
    mlp_title: str,
    attention_label: str,
    y_limits: tuple[float, float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
    y_min, y_max = y_limits

    max_abs = np.max(np.abs(potential)) if potential.size else 0.0
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
    return fig
