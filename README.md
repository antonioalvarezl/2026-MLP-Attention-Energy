# Transformer Dynamics as Wasserstein Gradient Flow

This repo simulates transformer-style dynamics (self-attention + MLP) on compact
manifolds, interpreted as Wasserstein gradient flows. Two geometries are supported:

- **S¹ (circle)**: dimension = 2
- **S² (sphere)**: dimension = 3

Two attention variants are available: unnormalized self-attention (USA) and
normalized self-attention (SA).

## Model

### S¹ (Circle)

Let $x_i(t) = (\cos \theta_i(t), \sin \theta_i(t)) \in S^1$ and
$t(\theta) = (-\sin \theta, \cos \theta)$ be the unit tangent. Define

$$
w_{ij} = \exp\bigl(\beta (x_i \cdot x_j - 1)\bigr)
       = \exp\bigl(\beta (\cos(\theta_i - \theta_j) - 1)\bigr).
$$

### S² (Sphere)

Let $x_i(t) \in S^2 \subset \mathbb{R}^3$ with $\|x_i\| = 1$. Define

$$
w_{ij} = \exp\bigl(\beta (x_i \cdot x_j - 1)\bigr).
$$

### MLP Drift

The MLP drift is a tangent vector field:

$$
u_{\mathrm{MLP}}(x) = \Pi_x \sum_{m=1}^k \omega_m\, \sigma(a_m \cdot x),
$$

where $a_m \in S^{d-1}$, $\omega_m \in \mathbb{R}^d$, $\sigma \in \{\mathrm{relu}, \mathrm{gelu}\}$,
and $\Pi_x$ is the projection onto the tangent space at $x$.

When `gradient_MLP=true`, we constrain $\omega_m = s_m a_m$ so the MLP is a gradient field.

### Dynamics

**Unnormalized self-attention (USA):**

$$
\dot{x}_i = s \cdot \frac{1}{N} \sum_{j=1}^N w_{ij} \Pi_{x_i}(x_j - x_i) + u_{\mathrm{MLP}}(x_i).
$$

**Normalized self-attention (SA):**

$$
\dot{x}_i = s \cdot \frac{\sum_{j=1}^N w_{ij} \Pi_{x_i}(x_j - x_i)}{\sum_{j=1}^N w_{ij}} + u_{\mathrm{MLP}}(x_i).
$$

Here $s = -1$ if `ascending=true` (gradient ascent) and $s = +1$ if
`ascending=false` (gradient descent).

For USA, the self-attention drift corresponds to the gradient flow of

$$
\mathsf{E}_\beta[\mu] = \frac{1}{2\beta} \iint e^{\beta x \cdot y}\, d\mu(x)\, d\mu(y).
$$

## Quick Start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Edit `config.json` with your settings.

See `CONFIG.md` for all configuration options and valid values.

3) Run

```bash
python3 main.py
```

## Outputs

Outputs are written under `results/experiment_YYYYMMDD_HHMMSS`. Each beta
produces its own run folder.

### Common outputs (S¹ and S²)

- `params.json` — full parameters and seeds
- `summary.json` — per-beta cluster counts for stats aggregation

### S¹ specific outputs

- `gamma_k.pdf` — eigenvalue spectrum
- `trajectories_null*.pdf` — null model (MLP=0) trajectories
- `trajectories_MLP*.pdf` — MLP trajectories
- `histogram*.pdf` — final particle distribution
- `evolution_*.gif` — animated evolution (if `gif_sphere=true`)

### S² specific outputs

- `sphere_init_*.pdf` — initial particle configuration
- `sphere_middle_*.pdf` — mid-evolution snapshot
- `sphere_final_*.pdf` — final configuration
- `sphere_histogram_*.pdf` — 3D bar histogram (φ-θ projection)
- `sphere_histogram_*_boundaries.pdf` — histogram with MLP decision boundaries
- `sphere_trajectory_*.pdf` — 3D trajectory plot (linear time scale)
- `sphere_trajectory_*_log.pdf` — 3D trajectory plot (log time scale)
- `sphere_evolution.gif` — animated sphere evolution (if `gif_sphere=true`)
- `sphere_histogram.gif` — animated histogram (if `gif_histogram=true`)
- `sphere_views.html` — interactive 3D visualization with:
  - MLP vs Null comparison
  - Initial/Middle/Final time states
  - Points vs Histogram display modes
  - Potential field overlay toggle
  - PDF/PNG export buttons

### Experiment-level outputs

At the experiment root:

- `stats/cluster_count.pdf` — cluster count vs √β (MLP only)
- `stats/cluster_count_with_null.pdf` — cluster count vs √β (MLP + null)
- `mlp_scale_sweep.json` — when `mlp_scale` is a list

## Resuming Experiments

If you cancel a run, you can resume by setting `experiment_dir` to the existing
experiment folder:

```json
"experiment_dir": "results/experiment_20260120_123456"
```

The system detects completed betas by matching `params.json` and skips them.
Stats are generated from all available `summary.json` files at the end.

## Memory Optimization

For large simulations (many particles, long runs), memory usage is optimized:

- Full histories are only saved when needed for GIFs or trajectory plots
- Set `gif_sphere=false`, `gif_histogram=false`, `pdf_trajectory=false` to minimize memory
- Memory is explicitly freed after each beta with garbage collection

## Repo Structure

```
main.py                 # Entry point; loads config and dispatches to d=2 or d=3
config.json             # User configuration
CONFIG.md               # Reference of all config options
requirements.txt        # Python dependencies
results/                # Output folder with experiment runs

wgf_circle/             # S¹ (d=2) implementation
  config.py             # Config parsing/validation
  runner.py             # Simulation loop and output generation
  dynamics.py           # Attention/MLP drift and integrators
  analysis.py           # Clustering metrics and convergence
  plotting.py           # Matplotlib plots and GIF generation
  io.py                 # Filesystem helpers and JSON utilities

wgf_sphere/             # S² (d=3) implementation
  config_S2.py          # Config parsing/validation
  runner_S2.py          # Simulation loop and output generation
  dynamics_S2.py        # Attention/MLP drift on S²
  analysis_S2.py        # Clustering on sphere (geodesic distance)
  plotting_S2.py        # 3D plots, histograms, interactive HTML
```

## Dependencies

- numpy, scipy — numerical computation
- matplotlib — static plots
- Pillow, imageio — GIF generation
- plotly (optional) — interactive HTML visualizations
