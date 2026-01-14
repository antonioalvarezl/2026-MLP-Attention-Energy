# Transformer Dynamics as Gradient Flow on the sphere

This repo simulates transformer-style dynamics (self-attention + MLP) on the
unit circle S1, interpreted as a gradient flow. Two variants are supported:
unnormalized self-attention (USA) and normalized self-attention (SA).

**Unnormalized self-attention + Gradient Ascent**

![Demo](examples/USA.gif)

**Normalized self-attention + Gradient Ascent**
![Demo](examples/SA.gif)

**Unnormalized self-attention + Gradient Descent**

![Demo](examples/USAd.gif)

**Normalized self-attention + Gradient Descent**

![Demo](examples/SAd.gif)

## Model

Let \(x_i(t) = (\cos \theta_i(t), \sin \theta_i(t)) \in S^1\) and
\(t(\theta) = (-\sin \theta, \cos \theta)\) be the unit tangent. Define

\[
w_{ij} = \exp\bigl(\beta (x_i \cdot x_j - 1)\bigr)
       = \exp\bigl(\beta (\cos(\theta_i - \theta_j) - 1)\bigr).
\]

The MLP drift is

\[
u_{\mathrm{MLP}}(\theta) = t(\theta) \cdot \sum_{m=1}^k \omega_m\, \sigma(a_m \cdot x(\theta)),
\]

where \(a_m \in S^1\), \(\omega_m = s_m a_m\) (gradient field), and
\(\sigma \in \{\mathrm{relu}, \mathrm{gelu}\}\).

The two dynamics are:

Unnormalized self-attention (USA):

\[
\dot{\theta}_i =
s \cdot \frac{1}{N} \sum_{j=1}^N w_{ij} \sin(\theta_i - \theta_j)
 + u_{\mathrm{MLP}}(\theta_i).
\]

Normalized self-attention (SA):

\[
\dot{\theta}_i =
s \cdot \frac{\sum_{j=1}^N w_{ij} \sin(\theta_i - \theta_j)}
{\sum_{j=1}^N w_{ij}}
 + u_{\mathrm{MLP}}(\theta_i).
\]

Here \(s = -1\) if `ascending=true` (gradient ascent) and \(s = +1\) if
`ascending=false` (gradient descent). In USA, setting `self_attention=false`
removes the \(j=i\) term. In SA, self-interaction is always included.

For USA, the self-attention drift corresponds to the gradient flow of

\[
\mathsf{E}_\beta[\mu] = \frac{1}{2\beta} \iint e^{\beta x \cdot y}\, d\mu(x)\, d\mu(y),
\]

up to a constant shift induced by \(x \cdot y - 1\).

## Quick start

1) Install dependencies

```
pip install -r requirements.txt
```

2) Edit `config.json` with your settings.

See `CONFIG.md` for all configuration options and valid values.

3) Run

```
python3 main.py
```

Outputs are written under `results/experiment_YYYYMMDD_HHMMSS`. Each beta
produces its own run folder with:
- `params.json` (full parameters and seeds)
- `summary.json` (per-beta counts used for stats)
- `gamma_k.pdf`
- `trajectories_null*.pdf` and `trajectories_null*_log.pdf`
- `trajectories_MLP*.pdf` and `trajectories_MLP*_log.pdf`
- `histogram*.pdf` (MLP only)
- `histogram_with_null*.pdf` (MLP + null)
- `total_clusters_beta=*.pdf` (when `num_point_inits > 1`)
- `mlp_scale_stop_time.pdf` (when `mlp_scale` is a list; saved in the run folder
  for the smallest scale per beta)
- `frames_MLP_null_init*` and `frames_MLP*_init*` (only `frame_first.pdf`,
  `frame_middle.pdf`, `frame_last.pdf`, plus `frame_*_noshade.pdf`)
- `evolution_MLP_null_init*.gif`
- `evolution_MLP*_init*.gif`
- `evolution_MLP_comparison*.gif`
- `field_MLP*_init*.gif`

At the experiment root:
- `mlp_scale_sweep.json` (when `mlp_scale` is a list)
- `stats/cluster_count.pdf`
- `stats/mass_count.pdf`
- `stats/mode_count.pdf`
- `stats/cluster_count_with_null.pdf`
- `stats/mass_count_with_null.pdf`
- `stats/mode_count_with_null.pdf`

Stats are only generated for single-scale runs; scale sweeps skip stats.

If a matching `params.json` already exists, that beta is skipped.

## Repo structure

- `main.py`: entry point; loads config and runs the experiment.
- `config.json`: user configuration.
- `CONFIG.md`: reference of all config options and valid values.
- `wgf/config.py`: config parsing/validation and seed planning.
- `wgf/runner.py`: simulation loop, output writing, stats aggregation.
- `wgf/dynamics.py`: attention/MLP drift and numerical integrators.
- `wgf/analysis.py`: clustering metrics and convergence detection.
- `wgf/plotting.py`: plots and GIF generation.
- `wgf/io.py`: filesystem helpers and JSON utilities.
- `requirements.txt`: Python dependencies.
- `results/`: output folder with experiment runs.
