"""Helpers for loading explicit MLP parameters from JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

VALID_ACTIVATIONS = {"relu", "gelu"}


def load_mlp_params_file(
    path: Path,
    dimension: int,
    default_activation: str,
) -> tuple[List[Tuple[np.ndarray, np.ndarray, str]], str]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    entries, base_activation = _extract_entries(data, default_activation)
    params: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("MLP params entries must be objects.")
        activation = entry.get("activation", base_activation)
        if activation is None:
            activation = base_activation
        activation = str(activation).strip().lower()
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f"Unsupported activation in MLP params: {activation}")
        a = _coerce_matrix("a", entry.get("a"), dimension)
        omega = _coerce_matrix("omega", entry.get("omega"), dimension)
        if a.shape != omega.shape:
            raise ValueError("MLP params require omega to have the same shape as a.")
        params.append((a, omega, activation))
    if not params:
        raise ValueError("MLP params file did not contain any parameter sets.")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return params, digest


def _extract_entries(
    data: Any,
    default_activation: str,
) -> tuple[list[dict], str]:
    base_activation = default_activation
    if isinstance(data, dict):
        if data.get("activation") is not None:
            base_activation = data["activation"]
        if "params" in data:
            entries = data["params"]
        elif "mlp_params" in data:
            entries = data["mlp_params"]
        elif "a" in data and "omega" in data:
            entries = [data]
        else:
            raise ValueError("MLP params file must contain 'a'/'omega' or 'params'.")
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("MLP params file must contain a JSON object or list.")
    if not isinstance(entries, list):
        raise ValueError("MLP params 'params' must be a list.")
    return entries, str(base_activation).strip().lower()


def _coerce_matrix(label: str, value: Any, dimension: int) -> np.ndarray:
    if value is None:
        raise ValueError(f"MLP params entry missing '{label}'.")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"MLP params '{label}' must be a 2D array.")
    if arr.shape[0] == 0:
        raise ValueError(f"MLP params '{label}' must be non-empty.")
    if arr.shape[1] != dimension:
        raise ValueError(f"MLP params '{label}' must have dimension {dimension}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"MLP params '{label}' must be finite.")
    return arr
