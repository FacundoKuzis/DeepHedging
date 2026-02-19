# Diffusion Models Workflow and Results (1b to 4b)

## 1) What We Are Doing
We are training diffusion-based simulators (DDPM) to generate synthetic 22-day price paths for deep hedging experiments.

The target is to match real market behavior better than GAN-like approaches, especially:
1. Heavy tails (extreme returns).
2. Dependence structure (especially squared-return autocorrelation, volatility clustering).
3. Path-risk shape (terminal distribution and drawdown behavior).

All runs use:
1. Asset: `^GSPC`.
2. Train split: `2005-01-01` to `2019-12-31`.
3. Test split: `2020-01-01` to `2024-12-31`.
4. Horizon: `n=22` trading days.
5. Same output root and artifact format used in the rest of the pipeline.

## 2) Step-by-Step Pipeline (DDPM)
1. Load a strict JSON config (`schema_version=1`, `model_family=diffusion`), no hidden defaults.
2. Resolve train/test market CSVs from cache (`market_data_cache`), or download if missing.
3. Build the raw return series from prices (`log_returns`).
4. Build feature channels (`feature_mode`):
1. `returns_only`, or
2. `returns_plus_abs_return`.
5. Transform channels into model space:
1. Main return channel uses `return_transform` (`minmax`, `gaussian_cdf`, `empirical_cdf`).
2. Extra channels are transformed consistently.
6. Build explicit rolling windows with configured `stride` and `min_windows`:
1. tensor shape `(n_windows, seq_len=22, n_features)`.
7. Train DDPM:
1. noise schedule (`beta_schedule`, `diffusion_steps`),
2. Conv1D residual denoiser,
3. MSE noise-prediction objective,
4. optional EMA weights.
8. Sample synthetic windows with reverse diffusion (`sampler_type`).
9. Inverse-transform synthetic return channel and reconstruct prices:
1. `P_0=S0`,
2. `P_t = S0 * cumprod(exp(log_return_t))`.
10. Evaluate against real windows (train and test splits):
1. legacy summary metrics,
2. return and squared-return dependence metrics,
3. tail quantile and exceedance metrics,
4. path-risk metrics.
11. Save all artifacts (`scores`, `plot_inputs`, plots, model files, schedule files, copied config).

## 3) Configs Tested (1b to 4b)
These runs are the same structural experiments as `diffusion_1..4` but with `train_epochs=500` and larger evaluation sample sizes.

| Run | Return Transform | Feature Mode | Diffusion Steps | Hidden / Blocks | Sampler | LR | Epochs |
|---|---|---|---:|---|---|---:|---:|
| diffusion_1b | minmax | returns_only | 200 | 128 / 4 | ddpm | 2e-4 | 500 |
| diffusion_2b | gaussian_cdf | returns_only | 200 | 128 / 4 | ddpm | 2e-4 | 500 |
| diffusion_3b | empirical_cdf | returns_only | 200 | 128 / 4 | ddpm | 2e-4 | 500 |
| diffusion_4b | gaussian_cdf | returns_plus_abs_return | 300 | 160 / 4 | ddpm | 2e-4 | 500 |

Common run-size parameters:
1. `n_synth_paths=1000`
2. `n_real_windows_compare=1000`
3. `hist_bins=50`

## 4) Training Behavior
From `training_loss_history.csv`:

| Run | Epochs Logged | Loss Start | Loss End | Loss Min |
|---|---:|---:|---:|---:|
| diffusion_1b | 500 | 0.4692 | 0.0460 | 0.0393 |
| diffusion_2b | 500 | 0.6092 | 0.1997 | 0.1888 |
| diffusion_3b | 500 | 0.5854 | 0.2235 | 0.2157 |
| diffusion_4b | 500 | 0.5699 | 0.0899 | 0.0822 |

Interpretation:
1. `1b` and `4b` fit the denoising objective more strongly.
2. `2b` and `3b` remain at higher loss, despite long training.

## 5) Test Results (Key Metrics)
Lower is better in all rows below.

| Run | daily_std_err | daily_kurt_err | terminal_hist_l1 | log_hist_l1 | dep_sq_mean_abs_err | dep_sq_lag1_err | tail_q0.001_err | tail_q0.01_err | left_exc_1pct_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diffusion_1b | 0.00554 | 2.93172 | 0.656 | 0.45182 | 0.03646 | 0.06038 | 0.04636 | 0.01081 | 0.07686 |
| diffusion_2b | 0.00376 | 3.73760 | 0.520 | 0.32027 | 0.02820 | 0.03663 | 0.04494 | 0.01239 | 0.03914 |
| diffusion_3b | 0.00216 | 25.84109 | 0.486 | 0.23773 | 0.02970 | 0.01872 | 0.00525 | 0.01106 | 0.03441 |
| diffusion_4b | 0.00339 | 3.00965 | 0.444 | 0.30709 | 0.01639 | 0.01759 | 0.04494 | 0.00912 | 0.04100 |

Additional path-risk mean absolute error:
1. `diffusion_1b`: `0.02187`
2. `diffusion_2b`: `0.01584`
3. `diffusion_3b`: `0.00967`
4. `diffusion_4b`: `0.01379`

## 6) Run-by-Run Interpretation
### diffusion_1b
1. `minmax` control is weakest overall.
2. Biggest problems: squared-return dependence and left-tail exceedance mismatch.

### diffusion_2b
1. Better global fit than `1b`.
2. Still weaker than `4b` on squared-return dependence and tails.

### diffusion_3b
1. Best one-step shape fit (`log_hist_l1`) and best extreme-left quantile (`q0.001`).
2. But kurtosis error is very large (`25.84`), signaling over-heavy synthetic tails / instability.
3. Not balanced enough for robust deployment.

### diffusion_4b
1. Most balanced run among `1b..4b`.
2. Best terminal histogram fit and best squared-return dependence block.
3. Good `q0.01` tail error.
4. Still weak on the most extreme left tail (`q0.001`).

## 7) Current Conclusion
Best current base candidate is `diffusion_4b` due to balance across:
1. terminal distribution,
2. squared-return dependence,
3. acceptable tail behavior (except extreme-left edge).

Main unresolved gap:
1. Extreme-left tail calibration (`q0.001`, left exceedance) remains off in the balanced models.

## 8) Files Used for This Analysis
All metrics were read from:
1. `G:/Mi unidad/Tesis2026/TimeGanTraining/diffusion_1b/csvs/scores/*`
2. `G:/Mi unidad/Tesis2026/TimeGanTraining/diffusion_2b/csvs/scores/*`
3. `G:/Mi unidad/Tesis2026/TimeGanTraining/diffusion_3b/csvs/scores/*`
4. `G:/Mi unidad/Tesis2026/TimeGanTraining/diffusion_4b/csvs/scores/*`

