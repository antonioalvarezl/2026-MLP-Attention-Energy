# Configuration

The simulator reads parameters from `config.json`. All keys are optional; the
defaults live in `wgf/config.py` under `DEFAULT_CONFIG`.

## Simulation

- betas: list of floats or comma-separated string; all values > 0.
- n_particles: int > 0.
- dt: float > 0.
- total_time: float > 0, or `"inf"`, `"infty"`, `"infinite"`, `"infinity"`.
- max_steps: int > 0; only used when `total_time` is infinite.
- save_every: int > 0; save one history snapshot every N steps (also used by
  `convergence_window`).
- integrator: `"euler"`, `"rk2"`, `"rk4"`.
- attention_mode: `"unnormalized"` or `"normalized"`.
- self_attention: bool; if `false`, removes the j=i term in USA. In SA the
  self-interaction is always included.
- ascending: bool; if `true`, flips the sign of the attention drift.
- dimension: must be 2 (S1).
- num_steps is derived from `total_time` and `dt` (or `max_steps` if
  `total_time=inf`).

## MLP

- mlp_units: int > 0 or `null`; `null` uses `dimension`.
- activation: `"relu"` or `"gelu"` (required).
- mlp_scale: float or list of floats; scale of MLP weights. If a list is
  provided, `betas` must contain a single value and each scale is run
  separately. If the list has three numbers `[start, stop, step]`, a range is
  generated from start to stop using the step size.
- mlp_scale_mode: `"std"` or `"norm"`; `"std"` treats `mlp_scale` as the
  standard deviation of the MLP weight scalars, `"norm"` fixes `|omega_j|`
  to `mlp_scale` with a random sign.
- gradient_MLP: bool; if `false`, the MLP is not constrained to be a gradient field
  and potential overlays are skipped in plots.
- num_mlp_inits: int > 0; number of independent MLP draws per beta.

## Initializations and seeds

- num_point_inits: int > 0; number of particle initializations per beta.
- particle_seed: int; base seed for particles (extra seeds are derived
  deterministically when `num_point_inits > 1`).
- mlp_seed: int; base seed for MLP (extra seeds are derived deterministically
  when `num_mlp_inits > 1`).

## Clustering and convergence

- k_max: int >= 0; max k for gamma and convergence plots.
- cluster_scale: float > 0; threshold = cluster_scale / sqrt(beta).
- mass_threshold: float in [0, 1]; minimum mass fraction for `mass_count`.
- convergence_window: int > 0; number of consecutive snapshots with the same
  cluster count to declare convergence.
- convergence_drift_tol: float >= 0; when `total_time` is infinite, requires
  max |drift| <= tol (0 disables).
- convergence_spread_factor: float >= 0; when `total_time` is infinite, requires
  max spread <= factor * threshold (0 disables).
- when `total_time` is infinite, the run stops at convergence or `max_steps`.

## Outputs

- results_dir: path; base output directory.
- experiment_dir: optional path; if set, outputs go there and are reused to
  continue betas. Existing betas are skipped and still counted for stats.
- gifs: bool; if false, skip generating GIFs (frames are still saved).
- plot_interval: float > 0; time spacing for GIF frames.
- output_frame_limit: int > 0; if exceeded, GIFs are skipped but
  `frame_first/middle/last` are still saved.
- when `mlp_scale` is a list, a `mlp_scale_sweep.json` file is written at the
  experiment root with per-scale summaries.
