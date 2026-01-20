"""Analysis utilities for convergence and clustering."""
from __future__ import annotations

import numpy as np

from .dynamics import TWO_PI


def cluster_threshold(beta: float, scale: float) -> float:
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    return scale / np.sqrt(beta)


def _circular_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.0:
        return values
    radius = int(np.ceil(3.0 * sigma_bins))
    if radius < 1:
        return values
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    padded = np.concatenate([values[-radius:], values, values[:radius]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def mode_count(theta: np.ndarray, threshold: float) -> int:
    """Count smoothed density modes (histogram-based)."""
    if theta.size == 0:
        return 0
    if threshold <= 0.0:
        return theta.size

    angles = theta % TWO_PI
    bin_target = max(threshold / 3.0, 1e-3)
    n_bins = max(60, int(np.ceil(TWO_PI / bin_target)))
    counts, edges = np.histogram(angles, bins=n_bins, range=(0.0, TWO_PI), density=True)
    bin_width = edges[1] - edges[0]
    sigma = max(threshold / 2.0, bin_width)
    sigma_bins = sigma / bin_width
    smoothed = _circular_smooth(counts, sigma_bins)

    peak_range = np.max(smoothed) - np.min(smoothed)
    if peak_range <= 0.05 * np.max(smoothed):
        return 1

    left = np.roll(smoothed, 1)
    right = np.roll(smoothed, -1)
    peak_mask = (smoothed >= left) & (smoothed >= right)
    height_thresh = 0.2 * np.max(smoothed)
    peaks = np.where(peak_mask & (smoothed >= height_thresh))[0]
    if peaks.size == 0:
        return 1

    min_sep_bins = max(1, int(np.ceil(threshold / bin_width)))
    peaks_sorted = sorted(peaks, key=lambda idx: smoothed[idx], reverse=True)
    selected = []
    for idx in peaks_sorted:
        if all(
            min((idx - chosen) % n_bins, (chosen - idx) % n_bins) >= min_sep_bins
            for chosen in selected
        ):
            selected.append(idx)
    return len(selected)


def _cluster_sizes(theta: np.ndarray, threshold: float) -> np.ndarray:
    if theta.size == 0:
        return np.array([], dtype=int)
    if threshold <= 0.0:
        return np.ones(theta.size, dtype=int)
    if threshold >= TWO_PI:
        return np.array([theta.size], dtype=int)

    angles = np.sort(theta % TWO_PI)
    gaps = np.diff(angles)
    wrap_gap = (angles[0] + TWO_PI) - angles[-1]
    gaps = np.concatenate([gaps, [wrap_gap]])
    start_idx = (int(np.argmax(gaps)) + 1) % angles.size
    linear = np.concatenate([angles[start_idx:], angles[:start_idx] + TWO_PI])

    sizes = []
    cluster_start = linear[0]
    count = 1
    for angle in linear[1:]:
        if angle - cluster_start <= threshold:
            count += 1
        else:
            sizes.append(count)
            cluster_start = angle
            count = 1
    sizes.append(count)
    return np.asarray(sizes, dtype=int)


def cluster_count(theta: np.ndarray, threshold: float) -> int:
    """Count clusters by greedy arc-length grouping on S1."""
    sizes = _cluster_sizes(theta, threshold)
    return int(sizes.size)


def cluster_max_spread(theta: np.ndarray, threshold: float) -> float:
    """Return the maximum within-cluster angular spread."""
    if theta.size == 0:
        return 0.0
    if threshold <= 0.0:
        return 0.0
    if threshold >= TWO_PI:
        return float(TWO_PI)

    angles = np.sort(theta % TWO_PI)
    gaps = np.diff(angles)
    wrap_gap = (angles[0] + TWO_PI) - angles[-1]
    gaps = np.concatenate([gaps, [wrap_gap]])
    start_idx = (int(np.argmax(gaps)) + 1) % angles.size
    linear = np.concatenate([angles[start_idx:], angles[:start_idx] + TWO_PI])

    max_spread = 0.0
    cluster_start = linear[0]
    cluster_end = linear[0]
    for angle in linear[1:]:
        if angle - cluster_start <= threshold:
            cluster_end = angle
        else:
            max_spread = max(max_spread, cluster_end - cluster_start)
            cluster_start = angle
            cluster_end = angle
    max_spread = max(max_spread, cluster_end - cluster_start)
    return float(max_spread)


def mass_count(theta: np.ndarray, threshold: float, mass_threshold: float) -> int:
    """Count clusters whose mass exceeds the given fraction threshold."""
    if mass_threshold <= 0.0:
        return cluster_count(theta, threshold)
    if mass_threshold >= 1.0:
        return 1 if theta.size > 0 else 0
    sizes = _cluster_sizes(theta, threshold)
    if sizes.size == 0:
        return 0
    min_size = mass_threshold * float(theta.size)
    return int(np.sum(sizes >= min_size))


def convergence_index(
    theta_hist: np.ndarray,
    threshold: float,
    window: int,
) -> int:
    total = theta_hist.shape[0]
    if total == 0:
        return 0
    if window <= 1:
        return total - 1

    counts = [cluster_count(theta_hist[idx], threshold) for idx in range(total)]
    for idx in range(total - 1, window - 2, -1):
        window_counts = counts[idx - window + 1 : idx + 1]
        if len(set(window_counts)) == 1:
            return idx
    return total - 1
