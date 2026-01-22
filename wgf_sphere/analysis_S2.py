"""Analysis utilities for clustering on S2."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def cluster_count_s2(points: np.ndarray, threshold: float) -> int:
    """Count clusters on S2 by geodesic distance connectivity."""
    if points.size == 0:
        return 0
    if threshold <= 0.0:
        return int(points.shape[0])
    if threshold >= np.pi:
        return 1

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    points = points / norms

    chord = 2.0 * np.sin(0.5 * threshold)
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=chord)

    visited = np.zeros(points.shape[0], dtype=bool)
    clusters = 0
    for idx in range(points.shape[0]):
        if visited[idx]:
            continue
        clusters += 1
        stack = [idx]
        visited[idx] = True
        while stack:
            current = stack.pop()
            for nbr in neighbors[current]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
    return clusters


def cluster_max_spread_s2(points: np.ndarray, threshold: float) -> float:
    """Return the maximum within-cluster geodesic spread on S2."""
    if points.size == 0:
        return 0.0
    if threshold <= 0.0:
        return 0.0
    if threshold >= np.pi:
        return float(np.pi)

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    points = points / norms

    chord = 2.0 * np.sin(0.5 * threshold)
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=chord)

    visited = np.zeros(points.shape[0], dtype=bool)
    max_spread = 0.0
    for idx in range(points.shape[0]):
        if visited[idx]:
            continue
        stack = [idx]
        visited[idx] = True
        cluster = []
        while stack:
            current = stack.pop()
            cluster.append(current)
            for nbr in neighbors[current]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        if len(cluster) <= 1:
            continue
        subset = points[cluster]
        dots = np.clip(subset @ subset.T, -1.0, 1.0)
        min_dot = float(np.min(dots))
        spread = float(np.arccos(min_dot))
        max_spread = max(max_spread, spread)
    return max_spread


def cluster_masses_s2(points: np.ndarray, threshold: float) -> np.ndarray:
    """Return the mass (fraction of particles) in each cluster on S2, sorted descending."""
    if points.size == 0:
        return np.array([])
    n = points.shape[0]
    if threshold <= 0.0:
        return np.ones(n) / n  # Each particle is its own cluster
    if threshold >= np.pi:
        return np.array([1.0])  # All particles in one cluster

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    points = points / norms

    chord = 2.0 * np.sin(0.5 * threshold)
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=chord)

    visited = np.zeros(n, dtype=bool)
    sizes = []
    for idx in range(n):
        if visited[idx]:
            continue
        stack = [idx]
        visited[idx] = True
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nbr in neighbors[current]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        sizes.append(size)
    masses = np.array(sizes, dtype=float) / n
    return np.sort(masses)[::-1]  # Descending order


def heaviest_cluster_mass_s2(points: np.ndarray, threshold: float) -> float:
    """Return the mass (fraction of particles) in the largest cluster on S2."""
    masses = cluster_masses_s2(points, threshold)
    return float(masses[0]) if masses.size > 0 else 0.0


def convergence_index_s2(points_hist: np.ndarray, threshold: float, window: int) -> int:
    total = points_hist.shape[0]
    if total == 0:
        return 0
    if window <= 1:
        return total - 1

    counts = [cluster_count_s2(points_hist[idx], threshold) for idx in range(total)]
    for idx in range(total - 1, window - 2, -1):
        window_counts = counts[idx - window + 1 : idx + 1]
        if len(set(window_counts)) == 1:
            return idx
    return total - 1
