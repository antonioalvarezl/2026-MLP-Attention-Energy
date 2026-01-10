# Transformer dynamics as a gradient flow on the circle

This repo simulates transformer-style dynamics (self-attention + MLP) on the unit
circle $\\mathbb{S}^1$, interpreted as a gradient flow. Two variants are supported:
unnormalized self-attention (USA) and normalized self-attention (SA).

**Unnormalized self-attention**

![Demo](examples/USA.gif)

**Normalized self-attention**
![Demo](examples/SA.gif)

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
- `figure_MLP*.pdf` and `figure_MLP*_log.pdf`
- `frames_MLP_null_init*` and `frames_MLP*_init*` (only `frame_first.pdf`,
  `frame_middle.pdf`, `frame_last.pdf`)
- `evolution_MLP_null_init*.gif`
- `evolution_MLP*_init*.gif`
- `evolution_MLP_comparison*.gif`
- `field_MLP*_init*.gif`
- `convergence.pdf` (or `convergence_MLP*_init*.pdf` for multiple inits)
- `convergence_log.pdf` (or `convergence_MLP*_init*_log.pdf`)

At the experiment root:
- `stats/cluster_count.pdf`
- `stats/mass_count.pdf`
- `stats/mode_count.pdf`
- `stats/cluster_count_with_null.pdf`
- `stats/mass_count_with_null.pdf`
- `stats/mode_count_with_null.pdf`

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
