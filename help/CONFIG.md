# Configuration Reference

The simulator reads parameters from `config.json`. All keys are optional; defaults
are defined in `wgf_circle/config.py` (d=2) and `wgf_sphere/config_S2.py` (d=3).

---

## Geometry

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension` | int | 2 | Manifold dimension: `2` for S¹ (circle), `3` for S² (sphere) |

---

## Simulation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `betas` | list or string | `[0.1, 0.5, 1, 2, 5, 7, 9, 12, 15, 25]` | Inverse temperature values. Can be a list `[0.1, 1, 10]` or comma-separated string `"0.1, 1, 10"` |
| `n_particles` | int > 0 | 1000 | Number of particles |
| `dt` | float > 0 | 5e-4 | Time step |
| `total_time` | float > 0 or `"inf"` | 20.0 | Total simulation time. Use `"inf"`, `"infty"`, `"infinite"`, or `"infinity"` for auto-convergence |
| `max_steps` | int > 0 | 200000 | Maximum steps (only used when `total_time="inf"`) |
| `save_every` | int > 0 | 10 | Save history snapshot every N steps |
| `integrator` | string | `"euler"` | Numerical integrator: `"euler"`, `"rk2"`, `"rk4"` |

---

## Attention Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_mode` | string | `"unnormalized"` | `"unnormalized"` (USA) or `"normalized"` (SA) |
| `self_attention` | bool | false | Include j=i term in USA. Forced on when `attention_mode="normalized"` |
| `ascending` | bool | false | If true, uses the raw attention+MLP field (gradient ascent). If false, flips the total field (gradient descent) |

---

## MLP Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mlp_units` | int or null | null | Number of MLP hidden units. `null` uses `dimension` |
| `activation` | string | `"relu"` | Activation function: `"relu"` or `"gelu"` |
| `mlp_scale` | float or list | 0.5 | Scale of MLP weights. If list, runs each scale separately (requires single beta) |
| `mlp_scale_mode` | string | `"std"` | `"std"`: mlp_scale is std of weights. `"norm"`: mlp_scale is ‖ωₘ‖ |
| `gradient_MLP` | bool | true | If true, constrain ωₘ = sₘaₘ (gradient field). If false, ωₘ is independent |
| `num_mlp_inits` | int > 0 | 1 | Number of independent MLP initializations per beta |
| `mlp_params_path` | string or null | null | Path to JSON file with explicit MLP parameters (see below) |

### MLP Parameters File Format

```json
{
  "activation": "relu",
  "a": [[1.0, 0.0], [0.0, 1.0]],
  "omega": [[0.5, 0.0], [0.0, -0.5]]
}
```

- `a`: shape `(n_units, dimension)` — input weight vectors (normalized to unit sphere)
- `omega`: shape `(n_units, dimension)` — output weight vectors
- For S² use 3 columns: `[[x, y, z], ...]`

---

## Initialization and Seeds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_point_inits` | int > 0 | 1 | Number of particle initializations per beta |
| `particle_seed` | int | 7 | Base seed for particle initialization |
| `mlp_seed` | int | 11 | Base seed for MLP weight initialization |

Seeds are derived deterministically when multiple initializations are used.

---

## Clustering and Convergence

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_max` | int ≥ 0 | 20 | Maximum k for gamma spectrum (S¹) and convergence plots |
| `cluster_scale` | float > 0 | 1.0 | Clustering threshold = `cluster_scale / sqrt(beta)` (capped at `π/(2d)`) |
| `mass_threshold` | float ∈ [0,1] | 0.0 | Minimum mass fraction for `mass_count` stats (cluster size ≥ `mass_threshold * n_particles`) |
| `convergence_window` | int > 0 | 5 | Consecutive snapshots with same cluster count to declare convergence |
| `convergence_drift_tol` | float ≥ 0 | 1e-3 | Max drift magnitude for convergence (0 disables) |
| `convergence_spread_factor` | float ≥ 0 | 1.0 | Max spread ≤ factor × threshold for convergence (0 disables) |

When `total_time="inf"`, simulation stops at convergence or `max_steps`.

Convergence is checked every `save_every` steps using three conditions:
(1) cluster count stable for `convergence_window`, (2) max within-cluster spread
≤ `convergence_spread_factor * threshold`, and (3) max drift magnitude
≤ `convergence_drift_tol` (if > 0). This is the same for MLP=0 descent; there
is no special "sanity cap" in the current runner.

---

## Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | string | `"results"` | Base output directory |
| `experiment_dir` | string or null | null | Specific experiment folder. If null, creates timestamped folder. Set to resume interrupted runs |
| `plot_interval` | float > 0 | 0.1 | Time interval between GIF frames |
| `output_frame_limit` | int > 0 | 400 | Maximum frames for GIFs. Exceeded → skip GIF generation |
| `mlp0_output_frame_limit` | int or null | null | Frame cap for MLP=0 GIFs (S² only, used when `gradient_MLP=true`; does not affect convergence) |

On S¹ runs, plots are grouped into subfolders inside each run directory:
`frames/`, `histograms/`, and `trajectories/`.

---

## GIF and Plot Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gif_sphere` | bool | true | Generate evolution GIFs (circle for S¹, sphere for S²) |
| `gif_histogram` | bool | true (S²) / false (S¹) | Generate histogram GIFs (S² only) |
| `pdf_trajectory` | bool | true | Generate trajectory PDF plots (S² only). Set false to save memory |
| `sphere_gif_rotations` | float | 1.0 | Number of full rotations during sphere GIF (S² only) |

### Memory Optimization

For large simulations, disable outputs you don't need:

```json
{
  "gif_sphere": false,
  "gif_histogram": false,
  "pdf_trajectory": false
}
```

This prevents storing full simulation histories (especially for S²), dramatically reducing memory usage.

---

## S² Specific Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sphere_html_view` | string | `"mlp"` | Default view in interactive HTML: `"mlp"` or `"null"` |

---

## MLP Scale Sweep

When `mlp_scale` is a list:

1. `betas` must contain exactly one value
2. Each scale is run separately
3. Results are saved in `mlp_scale_sweep.json`
4. A special format `[start, stop, step]` generates a range

Example:
```json
{
  "betas": [10],
  "mlp_scale": [0.1, 0.5, 1.0, 2.0]
}
```

Or using range notation:
```json
{
  "betas": [10],
  "mlp_scale": [0.1, 2.0, 0.1]
}
```
This generates scales from 0.1 to 2.0 in steps of 0.1.

---

## Example Configurations

### Basic S¹ run
```json
{
  "dimension": 2,
  "betas": [1, 5, 10],
  "n_particles": 500,
  "total_time": 10.0
}
```

### S² with convergence detection
```json
{
  "dimension": 3,
  "betas": [0.1, 1, 5, 10, 25],
  "n_particles": 1000,
  "total_time": "inf",
  "max_steps": 200000,
  "convergence_window": 5,
  "convergence_drift_tol": 0.001
}
```

### Memory-efficient S² (no GIFs)
```json
{
  "dimension": 3,
  "betas": [1, 5, 10, 25, 50],
  "n_particles": 1000,
  "total_time": "inf",
  "gif_sphere": false,
  "gif_histogram": false,
  "pdf_trajectory": false
}
```

### Resume interrupted experiment
```json
{
  "dimension": 3,
  "betas": [1, 5, 10, 25, 50],
  "experiment_dir": "results/experiment_20260120_123456"
}
```
Completed betas are detected and skipped automatically.
