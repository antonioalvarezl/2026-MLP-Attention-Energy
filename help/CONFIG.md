# Configuration Options

The simulator reads settings from `config.json`. All keys are optional; defaults
are in `wgf/config.py` under `DEFAULT_CONFIG`. Values below list allowed options
and any special behavior.

- betas: list of floats or comma-separated string; all values must be > 0.
- n_particles: int > 0.
- dt: float > 0.
- total_time: float > 0, or one of "inf", "infty", "infinite", "infinity".
- max_steps: int > 0; used only when total_time is infinite.
- save_every: int > 0; save one history snapshot every N steps.
- k_max: int >= 0; upper k used for gamma_k and convergence plots.
- num_mlp_inits: int > 0; number of independent MLP parameter draws.
- num_point_inits: int > 0; number of independent particle initializations.
- mlp_units: int > 0, or null; null means use `dimension`.
- activation: one of "relu" or "gelu" (required).
- mlp_scale: float; only used when mlp_scale_mode is "fixed".
- mlp_scale_mode: one of "fixed" or "exp_beta".
  If "exp_beta", scale is exp(beta) clipped at exp(12).
- dimension: must be 2 (this code is for S1).
- attention_mode: one of "unnormalized" or "normalized".
  If "normalized", mlp_scale_mode is forced to "fixed".
- unnormalized_scale_mode: one of "standard" or "minus_beta".
  Only used when attention_mode is "unnormalized".
  "minus_beta" uses exp(row_max - beta) as the global scale factor.
- integrator: one of "euler", "rk2", "rk4".
- particle_seed: int; base seed for particle initializations.
- mlp_seed: int; base seed for MLP initializations.
- results_dir: path; base output directory.
- experiment_dir: optional path; if null/empty, a timestamped folder is created.
  If relative with no parent, it is created under results_dir; otherwise it is
  used as provided (absolute or relative to the current working directory).
- plot_interval: float > 0; spacing in time for GIF frames.
- cluster_scale: float > 0; cluster threshold is cluster_scale / sqrt(beta).
- mass_threshold: float in [0, 1].
- convergence_window: int > 0; window size for convergence detection.
- exclude_self: bool; ignored when attention_mode is "normalized".
- convergence_drift_tol: float >= 0; only used when total_time is infinite.
- convergence_spread_factor: float >= 0; only used when total_time is infinite.
- output_frame_limit: int > 0; max frames before skipping GIF creation.

Notes:
- num_steps is derived from total_time and dt (or max_steps for total_time=inf).
- tie_potential is always true in code and is not configurable.
