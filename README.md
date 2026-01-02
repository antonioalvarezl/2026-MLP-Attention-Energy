# Self-Attention + MLP Drift (S1) Simulations

This repo generates Figure-2-style plots for S1 particle dynamics with
self-attention and a fixed MLP drift term.

## Quick start

1) Install deps

```
pip install -r requirements.txt
```

2) Edit `config.json` with your settings.

See `CONFIG.md` for all configuration options and valid values.

3) Run the simulation

```
python3 main.py
```

Outputs are written under a timestamped experiment folder inside `results/`
named like `experiment_YYYYMMDD_HHMMSS`.
Each beta produces its own run folder with:
- `params.json` (full parameters and seeds)
- `summary.json` (per-beta counts used for stats plots)
- `figure_MLP*.png` (one per MLP initialization, 3x3 layout)
- `figure_MLP*_log.png` (same as above, with log-scaled time)
- `frames_MLP_null_init*` and `frames_MLP*_init*` (only `frame_first.png`, `frame_middle.png`, `frame_last.png`)
- `evolution_MLP*_init*.gif`
- `evolution_MLP_comparison*.gif` (left: MLP=0, right: std(MLP))
- `field_MLP*_init*.gif` (total field plus attention/MLP components)
- `convergence.png` (or `convergence_MLP*_init*.png` when multiple inits)
- `convergence_log.png` (or `convergence_MLP*_init*_log.png` when multiple inits)
At the root of the experiment folder, summary plots are generated:
- `stats/cluster_count.png`
- `stats/mass_count.png`
- `stats/mode_count.png`
- `stats/cluster_count_with_null.png`
- `stats/mass_count_with_null.png`
- `stats/mode_count_with_null.png`

If a matching `params.json` already exists, the run is skipped.

## Notes
- `particle_seed` and `mlp_seed` in `config.json` deterministically fix both particle and MLP initializations. With `num_mlp_inits = 1` and unchanged hyperparameters, the MLP parameters are identical across runs.
- `convergence_window` controls how many saved frames must keep the same cluster count to declare convergence.
- By default the MLP has one layer with `d` units (here `d=2`) and is constrained to be a gradient field.
- Use `plot_interval` in `config.json` to set the frame spacing for the evolution GIFs.
- The left panel shows the MLP-null density evolution with kmax noted in the title.
- Use `attention_mode = "normalized"` in `config.json` to switch to normalized self-attention.
- Use `integrator = "rk2"` or `"rk4"` in `config.json` for higher-order time stepping (default is `"euler"`).
- Set `total_time` to `"inf"` to run until convergence; `max_steps` caps the run.
- `exclude_self = true` removes self-interactions in attention (ignored when `attention_mode = "normalized"`). `convergence_drift_tol` and `convergence_spread_factor` control the auto-stop criteria for `"inf"`.
- `output_frame_limit` skips GIF generation when too many frames would be produced.
- Set `mlp_scale_mode = "exp_beta"` to use `exp(beta)` (clipped at `exp(12)`) for each beta.
- Set `experiment_dir` in `config.json` to reuse an existing experiment folder and regenerate `stats/` using already-finished betas.
- If `attention_mode = "normalized"`, the code forces `mlp_scale_mode = "fixed"`.
- `activation` must be `relu` or `gelu`.
- `unnormalized_scale_mode` controls the global scaling in unnormalized attention
  (`standard` vs `minus_beta`) and is ignored when `attention_mode = "normalized"`.

## Repo structure
- `main.py`: entry point; loads config and runs the experiment.
- `config.json`: user configuration (inputs for a run).
- `CONFIG.md`: reference of all config options and valid values.
- `wgf/config.py`: config parsing/validation and seed planning.
- `wgf/runner.py`: main simulation loop, output writing, stats aggregation.
- `wgf/dynamics.py`: attention/MLP drift and numerical integrators.
- `wgf/analysis.py`: clustering metrics and convergence detection.
- `wgf/plotting.py`: all plot/GIF generation utilities.
- `wgf/io.py`: filesystem helpers and JSON utilities.
- `requirements.txt`: Python dependencies.
- `results/`: output folder with experiment runs.
