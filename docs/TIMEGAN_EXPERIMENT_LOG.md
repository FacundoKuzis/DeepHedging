# TimeGAN Deep Hedging Experiment Log (Expert Review)

## 1) Objective and Scope
This document records the exact technical pipeline implemented in this repository to train and evaluate a TimeGAN-based synthetic price simulator for Deep Hedging, including:
- End-to-end data flow from JSON config to final artifacts
- Exact transformation stages and formulas
- What has been tested so far (`baseline_1` ... `baseline_10`)
- What works, what does not, and what is still missing

Code scope covered:
- `examples/timegan_simulator_console.py`
- `src/DeepHedging/HedgingInstruments/timegan_stock.py`
- `src/DeepHedging/utils/market_data.py`

---

## 2) End-to-End Data Lineage (What Happens to Data, Step by Step)

### 2.1 Entry Point and Config Name Resolution
Execution command:
- `python examples/timegan_simulator_console.py <config_name_optional>`

Resolution logic:
1. If `<config_name>` is missing, script prompts in console.
2. Only filename is accepted (no directory path).
3. If filename does not end in `.json`, the suffix is auto-appended.
4. File is loaded from `./gan_training_configs/<name>.json`.

No implicit defaults are injected. The run fails if anything required is missing/invalid.

---

### 2.2 Strict Config Validation Contract
The config must match exactly the required key set:
- `s0`, `t`, `n`, `r`
- `ticker`
- `train_start_date`, `train_end_date`
- `test_start_date`, `test_end_date`
- `interval`, `price_col`
- `download_if_missing`, `retrain`
- `stride`, `min_windows`
- `train_epochs`, `batch_size`, `noise_dim`, `layers_dim`, `latent_dim`
- `learning_rate`, `gamma`, `random_seed`
- `training_target`
- `match_return_moments`
- `input_return_clip_quantiles`, `output_return_clip_quantiles`
- `n_synth_paths`, `n_real_windows_compare`, `n_plot_paths`
- `rolling_vol_window`, `acf_max_lag`, `hist_bins`
- `normalization_base`

Validation rules enforced:
1. Missing keys -> error.
2. Unknown extra keys -> error.
3. Nullable keys allowed only for quantiles (`null`).
4. Strict type checks:
   - booleans for flags,
   - positive ints for counts/hyperparameters,
   - numeric for continuous params.
5. `training_target` must be `log_returns` or `price_levels`.
6. Quantiles must satisfy `0 <= low < high <= 1`.
7. Dates parsed as `%Y-%m-%d`, start <= end for train and test.
8. `acf_max_lag < n`.

---

### 2.3 Run Namespace and Output Directories
Global output root is fixed:
- `G:/Mi unidad/Tesis2026/TimeGanTraining`

For run name `<run_name>` (JSON stem), script creates:
- `.../<run_name>/plots`
- `.../<run_name>/csvs`
- `.../<run_name>/csvs/plot_inputs`
- `.../<run_name>/csvs/scores`
- `.../<run_name>/models`

Shared cache (across all runs):
- `.../market_data_cache`

Before training starts, the validated config JSON is copied into run folder:
- `.../<run_name>/<original_config_filename>.json`

This guarantees run reproducibility with the exact config snapshot used.

---

### 2.4 Train/Test Market Data Acquisition and Cache
The runner constructs cache file names:
- Train CSV: `<ticker>_<train_start>_<train_end>_<interval>.csv`
- Test CSV: `<ticker>_<test_start>_<test_end>_<interval>.csv`

If cache file exists, it is reused. If not, `TimeGANStock` calls market download utility.

Download process (`market_data.py`):
1. Primary source: `yfinance.download` (retries).
2. Secondary yfinance fallback: `yf.Ticker(...).history`.
3. If both fail/empty: Stooq fallback.

Normalization during ingestion:
1. Flatten possible MultiIndex columns from yfinance.
2. Normalize OHLCV names to canonical form (`Date/Open/High/Low/Close/Adj Close/Volume`).
3. Convert `Date` to datetime, numeric coercion for OHLCV, drop invalid rows.
4. Filter to requested `[start_date, end_date]`.
5. Sort by `Date` ascending.

S&P index alias logic in Stooq fallback:
- `^GSPC` -> tries `^spx`, `spx.us`, `^gspc`.

Important behavior:
- If source does not fully cover requested date range, a warning is printed with actual available min/max dates.

---

### 2.5 `TimeGANStock` Initialization on Train Split
When constructing train instrument:
1. Validate core hyperparameters (`N`, `stride`, `min_windows` > 0).
2. Resolve `csv_path` and `model_path`.
3. Load cached CSV or download then load.
4. Extract selected `price_col`, force numeric finite positive values.

Then internal data objects are built:
1. `self._series`: cleaned 1D price series.
2. `self._real_windows`: rolling windows of length `N+1` using configured `stride`.
3. `self._log_returns`: `diff(log(price))`.

Hard checks:
- Need enough observations for windows and returns.
- Need at least `min_windows` clean windows for training.

---

### 2.6 Transformation Logic Before Training
Branch by `training_target`:
- `log_returns` (current preferred path)
- `price_levels` (legacy compatibility path)

For `log_returns`:
1. Optional input clipping:
   - If `input_return_clip_quantiles` is set, clip return series to those quantiles.
2. Compute return moments on training input:
   - `mean`, `std` (with floor `1e-8`).
3. Optional output clip boundaries:
   - If `output_return_clip_quantiles` set, store low/high thresholds from training returns.
4. Store min/max for inverse scaling:
   - `returns_train_min`, `returns_train_max`.

For legacy `price_levels`:
- store `price_train_min`, `price_train_max`.

Training capacity check:
- Available sequences = `len(series_for_target) - sequence_length + 1`
- Must be `>= min_windows`.

Sequence length:
- `N` if `training_target=log_returns`
- `N+1` if `training_target=price_levels`

---

### 2.7 Model Load vs Train Decision
`_fit_or_load_synthesizer()` flow:
1. If model already loaded in memory, reuse.
2. If model file exists and `retrain=false`, load from disk.
3. Else train new model and save to disk.

Training uses `ydata-synthetic`:
- `TimeSeriesSynthesizer(modelname="timegan", model_parameters=...)`
- `ModelParameters`: `batch_size`, `lr`, `noise_dim`, `layers_dim`, `latent_dim`, `gamma`
- `TrainParameters`: `epochs`, `sequence_length`, `number_sequences=1`
- Input dataframe is single-column: `feature`

Random seeds are set (`numpy`, `tensorflow`) when provided.

Known limitation:
- Current API path does not expose per-epoch joint losses in a callback/history structure.

---

### 2.8 Sampling Synthetic Sequences and Price Reconstruction
Sampling stage (`generate_paths`):
1. Sample `num_paths` windows from synthesizer.
2. Validate shape and sequence length.

Postprocess branch A (preferred, `seq_len=N` returns):
1. Inverse min-max scaling from `[0,1]` back to return scale using train min/max.
2. Optional moment matching (if `match_return_moments=true`):
   - Affine transform synthetic returns to training mean/std.
3. Optional output quantile clipping (if configured).
4. Safety clip returns to `[-1, 1]`.
5. Convert returns to prices:
   - `gross = exp(log_return)`
   - `P_0 = S0`
   - `P_t = S0 * cumprod(gross)` for `t=1..N`
6. Enforce positivity and anchor first point exactly at `S0`.

Postprocess branch B (legacy, `seq_len=N+1` price levels):
1. Inverse min-max scaling on levels.
2. Re-anchor each path by first value:
   - `P_scaled = S0 * (P / P_0)`
3. Enforce positivity and set first point to `S0`.

Final output:
- `tf.Tensor` shape `(num_paths, N+1)`, dtype `float32`.

---

### 2.9 Train/Test Comparison Dataset Construction
After synthetic generation:
1. Runner instantiates a second `TimeGANStock` on test date range with same model path (`retrain=false`).
2. Samples real rolling windows from:
   - Train instrument (`real_train_paths`)
   - Test instrument (`real_test_paths`)
3. Synthetic paths from train model are compared against both splits.

This produces:
- In-sample diagnostic (`train split`)
- Out-of-sample diagnostic (`test split`)

---

### 2.10 Path Normalization for Comparable Plots/Metrics
All path sets are normalized to a common base:
- `normalized = normalization_base * path / path[:,0]`
- First value forced exactly to `normalization_base`.

This removes level-scale differences and focuses comparison on dynamics/shape.

---

### 2.11 Metrics Computed
From normalized paths, script computes:
1. One-step simple returns:
   - `r_t = P_t / P_{t-1} - 1`
2. One-step log returns:
   - `lr_t = log(P_t / P_{t-1})`
3. Terminal returns:
   - `R_T = P_T / P_0 - 1`
4. Skewness and excess kurtosis on one-step simple returns.
5. Rolling volatility distribution from one-step log returns.
6. Mean ACF over paths for lags `1..acf_max_lag`.
7. Histogram L1 distances (real vs synthetic) for:
   - terminal returns
   - one-step log returns

Histogram policy:
- Same bin edges are used for real and synthetic within each metric comparison.
- This applies both to plotted histograms and CSV histogram-density exports.

---

### 2.12 Plot Outputs (Per Split, JPG)
For each split (`train`, `test`), saved under:
- `.../<run>/plots/<split>/`

Files:
- `overlay_paths_normalized.jpg`
- `terminal_returns_hist.jpg`
- `one_step_log_returns_hist.jpg`
- `rolling_volatility_hist.jpg`
- `acf_lags.jpg`

---

### 2.13 CSV Outputs (Per Split + Scores)
Per split under:
- `.../<run>/csvs/plot_inputs/<split>/`

Saved datasets:
- `real_paths_normalized.csv`
- `synthetic_paths_normalized.csv`
- `real_one_step_simple_returns.csv`
- `synthetic_one_step_simple_returns.csv`
- `real_one_step_log_returns.csv`
- `synthetic_one_step_log_returns.csv`
- `real_terminal_returns.csv`
- `synthetic_terminal_returns.csv`
- `real_rolling_volatility_values.csv`
- `synthetic_rolling_volatility_values.csv`
- `acf_values.csv`
- `hist_terminal_returns.csv`
- `hist_one_step_log_returns.csv`
- `hist_rolling_volatility.csv`
- `summary_metrics.csv`

Run-level score files:
- `.../<run>/csvs/summary_metrics_all_splits.csv`
- `.../<run>/csvs/summary_metrics.csv` (legacy test-only summary)
- `.../<run>/csvs/scores/summary_metrics_train.csv`
- `.../<run>/csvs/scores/summary_metrics_test.csv`
- `.../<run>/csvs/scores/training_scores_available.csv`

`training_scores_available.csv` includes:
- cache/model existence flags before/after run
- wallclock generation time
- requested training hyperparameters
- estimated sigma from train series
- train window and observation counts
- explicit note that per-epoch joint losses are not exposed by current API path

---

## 3) Compact Dataflow Diagram
```text
JSON config
  -> strict schema/type/date validation
  -> run namespace creation + config copy
  -> market cache resolution (train/test CSV paths)
      -> if missing: yfinance download -> fallback history -> fallback Stooq
      -> normalize/filter/save CSV
  -> train TimeGANStock init
      -> load prices -> clean series -> rolling windows -> returns
      -> optional input clipping -> stats/minmax setup
      -> load model or fit model -> save model
  -> sample synthetic sequences
      -> inverse scaling -> optional moment match/clip -> returns->prices
  -> test TimeGANStock init (real windows only, same model path)
  -> normalize paths to base
  -> compute metrics + plots + hist-density tables
  -> save all split CSVs + run score CSVs + plot JPGs
```

---

## 4) Experiments Already Executed (High-Level)

### 4.1 Batch A (`baseline_1` to `baseline_5`)
Purpose:
- Explore clipping + moment-matching branch with varied gamma/stride/capacity/learning-rate.

Observed:
- Trade-off between terminal behavior and one-step distribution realism.
- Clipping/moment matching can stabilize some metrics but distort tails and shape.

### 4.2 Batch B (`baseline_6` to `baseline_10`)
Constraints imposed:
- `match_return_moments = false`
- `input_return_clip_quantiles = null`
- `output_return_clip_quantiles = null`

Purpose:
- Evaluate raw TimeGAN output quality without post-hoc distribution forcing.

Observed:
- Better methodological purity and easier interpretation.
- Tail mismatch (kurtosis) and lag-1 autocorrelation error remain major issues.

Current best candidate in this branch:
- `baseline_9` (most balanced terminal + temporal behavior among 6..10).

### 4.3 Detailed Config Matrix (Executed Runs)
Common fixed settings in all runs listed below:
- ticker/date/frequency: `^GSPC`, train `2005-01-01..2019-12-31`, test `2020-01-01..2024-12-31`, `interval=1d`
- horizon: `n=22`, `t=22/252`, `s0=100`, `r=0.05`
- evaluation sizing: `n_synth_paths=500`, `n_real_windows_compare=500`, `n_plot_paths=25`
- diagnostics: `rolling_vol_window=10`, `acf_max_lag=10`, `hist_bins=50`, `normalization_base=100`

Executed config deltas:

| Run | Seed | Epochs | Batch | LR | Gamma | Layers/Latent/Noise | Stride | Min windows | Match moments | Input clip q | Output clip q |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|
| baseline | 42 | 1000 | 128 | 2e-4 | 0.50 | 128/24/32 | 1 | 300 | False | None | None |
| baseline_1 | 43 | 1000 | 128 | 2e-4 | 0.50 | 128/24/32 | 1 | 300 | True | None | None |
| baseline_2 | 44 | 1000 | 128 | 2e-4 | 0.30 | 128/24/32 | 1 | 300 | True | [0.01, 0.99] | [0.01, 0.99] |
| baseline_3 | 45 | 1200 | 128 | 2e-4 | 0.30 | 128/24/32 | 3 | 200 | True | None | None |
| baseline_4 | 46 | 1800 | 128 | 1e-4 | 0.30 | 128/24/32 | 1 | 300 | True | [0.005, 0.995] | [0.005, 0.995] |
| baseline_5 | 47 | 1400 | 256 | 1.5e-4 | 0.20 | 96/16/24 | 2 | 250 | True | [0.01, 0.99] | [0.01, 0.99] |
| baseline_6 | 106 | 1400 | 128 | 1.5e-4 | 0.25 | 128/24/32 | 2 | 300 | False | None | None |
| baseline_7 | 107 | 1600 | 128 | 1.5e-4 | 0.25 | 128/24/32 | 3 | 300 | False | None | None |
| baseline_8 | 108 | 1600 | 256 | 1e-4 | 0.15 | 128/24/32 | 2 | 300 | False | None | None |
| baseline_9 | 109 | 1400 | 128 | 1.5e-4 | 0.25 | 192/32/32 | 2 | 300 | False | None | None |
| baseline_10 | 110 | 1200 | 256 | 2e-4 | 0.20 | 96/16/24 | 2 | 300 | False | None | None |

### 4.4 Detailed Metric Results (Test Split)
Metric source for each run:
- `G:/Mi unidad/Tesis2026/TimeGanTraining/<run>/csvs/scores/summary_metrics_test.csv`

Derived error view used for comparability (lower is better):
- `daily_std_err = |std_syn - std_real|`
- `daily_skew_err = |skew_syn - skew_real|`
- `daily_kurt_err = |kurt_syn - kurt_real|`
- `term_mean_err = |term_mean_syn - term_mean_real|`
- `term_std_err = |term_std_syn - term_std_real|`
- `term_hist_l1`, `log_hist_l1`, `acf_mean_err`, `acf_lag1_err` directly from summary CSV
- `composite` = mean of min-max normalized errors above (internal ranking indicator only)

| Run | daily_std_err | daily_skew_err | daily_kurt_err | term_mean_err | term_std_err | term_hist_l1 | log_hist_l1 | acf_mean_err | acf_lag1_err | composite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_3 | 0.001564 | 0.231296 | 12.315620 | 0.004853 | 0.028110 | 0.676000 | 0.424545 | 0.208167 | 0.775843 | 0.262433 |
| baseline_9 | 0.002773 | 0.504507 | 14.377084 | 0.007261 | 0.028738 | 0.732000 | 0.265636 | 0.193787 | 0.723786 | 0.345109 |
| baseline_2 | 0.004707 | 1.728611 | 8.391474 | 0.000171 | 0.003999 | 0.600000 | 0.958545 | 0.226047 | 0.371819 | 0.352037 |
| baseline_5 | 0.003182 | 0.401946 | 11.981864 | 0.000015 | 0.065805 | 1.028000 | 0.341091 | 0.242214 | 0.436242 | 0.359911 |
| baseline_1 | 0.001443 | 0.525616 | 12.238567 | 0.001469 | 0.064060 | 0.908000 | 0.278909 | 0.275592 | 0.841773 | 0.381666 |
| baseline | 0.002558 | 0.661056 | 13.258749 | 0.014145 | 0.036462 | 0.712000 | 0.320364 | 0.242918 | 0.820783 | 0.423982 |
| baseline_7 | 0.001903 | 1.122953 | 11.918016 | 0.008289 | 0.065913 | 0.968000 | 0.367455 | 0.266948 | 0.841114 | 0.495555 |
| baseline_4 | 0.003352 | 0.267282 | 14.186410 | 0.000111 | 0.061643 | 0.968000 | 0.460000 | 0.290757 | 0.876988 | 0.509581 |
| baseline_6 | 0.003108 | 0.720663 | 13.967066 | 0.007533 | 0.060294 | 0.896000 | 0.232000 | 0.289802 | 0.875410 | 0.516709 |
| baseline_10 | 0.002324 | 0.278974 | 13.500256 | 0.021176 | 0.079091 | 0.960000 | 0.287091 | 0.326687 | 0.878875 | 0.585707 |
| baseline_8 | 0.002524 | 0.701786 | 14.428964 | 0.002669 | 0.099956 | 1.104000 | 0.247273 | 0.394254 | 0.942984 | 0.643534 |

### 4.5 Per-Run Snapshots (Real vs Synthetic, Test Split)
Key raw values (from `summary_metrics_test.csv`) to understand directionality:

- `baseline_3`: `daily_std real/syn=0.013157/0.011593`, `daily_kurt real/syn=11.454914/-0.860706`, `term_std real/syn=0.056764/0.084874`, `acf_lag1_err=0.775843`, `log_hist_l1=0.424545`, `term_hist_l1=0.676000`.
- `baseline_9`: `daily_std real/syn=0.013627/0.010854`, `daily_kurt real/syn=13.788663/-0.588421`, `term_std real/syn=0.057787/0.086525`, `acf_lag1_err=0.723786`, `log_hist_l1=0.265636`, `term_hist_l1=0.732000`.
- `baseline_2`: `daily_std real/syn=0.013765/0.009058`, `daily_kurt real/syn=13.240713/4.849239`, `term_std real/syn=0.051751/0.055750`, `acf_lag1_err=0.371819`, `log_hist_l1=0.958545`, `term_hist_l1=0.600000`.
- `baseline_5`: `daily_std real/syn=0.013311/0.010129`, `daily_kurt real/syn=12.492557/0.510694`, `term_std real/syn=0.056651/0.122457`, `acf_lag1_err=0.436242`, `log_hist_l1=0.341091`, `term_hist_l1=1.028000`.
- `baseline_1`: `daily_std real/syn=0.013048/0.011605`, `daily_kurt real/syn=11.771087/-0.467479`, `term_std real/syn=0.053468/0.117528`, `acf_lag1_err=0.841773`, `log_hist_l1=0.278909`, `term_hist_l1=0.908000`.
- `baseline`: `daily_std real/syn=0.013539/0.010981`, `daily_kurt real/syn=12.529352/-0.729397`, `term_std real/syn=0.057134/0.093596`, `acf_lag1_err=0.820783`, `log_hist_l1=0.320364`, `term_hist_l1=0.712000`.
- `baseline_7`: `daily_std real/syn=0.013189/0.011286`, `daily_kurt real/syn=11.914258/-0.003758`, `term_std real/syn=0.058173/0.124086`, `acf_lag1_err=0.841114`, `log_hist_l1=0.367455`, `term_hist_l1=0.968000`.
- `baseline_4`: `daily_std real/syn=0.014004/0.010652`, `daily_kurt real/syn=14.008287/-0.178123`, `term_std real/syn=0.056756/0.118399`, `acf_lag1_err=0.876988`, `log_hist_l1=0.460000`, `term_hist_l1=0.968000`.
- `baseline_6`: `daily_std real/syn=0.013746/0.010638`, `daily_kurt real/syn=13.685444/-0.281623`, `term_std real/syn=0.057042/0.117336`, `acf_lag1_err=0.875410`, `log_hist_l1=0.232000`, `term_hist_l1=0.896000`.
- `baseline_10`: `daily_std real/syn=0.013551/0.011227`, `daily_kurt real/syn=12.999632/-0.500624`, `term_std real/syn=0.058758/0.137850`, `acf_lag1_err=0.878875`, `log_hist_l1=0.287091`, `term_hist_l1=0.960000`.
- `baseline_8`: `daily_std real/syn=0.013519/0.010995`, `daily_kurt real/syn=14.005114/-0.423850`, `term_std real/syn=0.057501/0.157457`, `acf_lag1_err=0.942984`, `log_hist_l1=0.247273`, `term_hist_l1=1.104000`.

### 4.6 Detailed Readout from the New Tables
1. No run solves tails yet:
   - `daily_kurt_err` remains high in all runs (best: `baseline_2` at `8.391474`, many around `12-14+`).
2. Lag-1 dependence is a major unresolved gap:
   - most runs have `acf_lag1_err ~0.72-0.94`, except `baseline_2` (`0.371819`) and `baseline_5` (`0.436242`).
3. Distribution-shape vs dependence trade-off is explicit:
   - `baseline_2` improves terminal std and lag1 but severely worsens one-step histogram (`log_hist_l1=0.958545`).
4. Among no-clip/no-moment runs (`baseline_6..10`), `baseline_9` remains strongest composite:
   - better `acf_mean_err` and `term_std_err` balance than peers.
5. `baseline_3` is best global composite in executed set, but it uses moment matching:
   - this makes it not directly aligned with the current constraint of avoiding moment forcing.

---

## 5) What Has Been Achieved
1. Full reproducible runner with strict config contract and deterministic artifact storage.
2. Robust market data layer with retry/fallback and persistent caching.
3. Integrated TimeGAN instrument compatible with DeepHedging instrument interface (`generate_paths`).
4. Split-aware evaluation framework (train/test) with comprehensive CSV and plot outputs.
5. Standardized histogram-bin policy across metrics and visual outputs.

---

## 6) What Is Still Missing
1. Tail behavior remains underfit:
   - synthetic excess kurtosis still far from real.
2. Temporal dependence mismatch:
   - lag-1 ACF error remains high in many runs.
3. Limited training introspection:
   - no native per-epoch losses from current `ydata-synthetic` integration path.
4. Calibration objective is still indirect:
   - model selection relies on post-hoc metrics, not on explicit train-time penalties for ACF/tails.

---

## 7) Recommended Next Technical Moves
1. Continue around `baseline_9` neighborhood (gamma/capacity/epochs grid).
2. Keep `no clipping` and `no moment matching` as requested for clean attribution.
3. Add optional external monitoring loop:
   - periodic checkpoint sampling + interim metrics (without changing core training API).
4. Evaluate alternative return representations:
   - volatility-standardized returns, regime-segmented training windows.

---

## 8) Reproducibility Commands
- `python examples/timegan_simulator_console.py baseline_9`
- `python examples/timegan_simulator_console.py baseline_9.json`

Artifacts:
- `G:/Mi unidad/Tesis2026/TimeGanTraining/<run_name>/...`

---

## 9) TimeGAN v2 Protocol (Urgent Additions)

### 9.1 Core v2 Changes
1. Explicit-window fitting:
   - Training tensor is now built in-repo with shape `(n_windows, seq_len, n_features)` when `fit_input_mode="explicit_windows"`.
   - This bypasses the internal stride-1 segmentation path from `ydata-synthetic` and preserves configured `stride`.
2. Return transforms:
   - Supported transforms: `minmax`, `gaussian_cdf`, `empirical_cdf`.
   - For `gaussian_cdf` and `empirical_cdf`, numerical clipping is applied only in `u`-space via `transform_eps`.
   - Legacy return hard-clip is now explicitly controlled by `legacy_return_clip`.
3. Feature channels:
   - `returns_only`
   - `returns_plus_abs_return`
   - `returns_plus_rolling_vol`
4. Diagnostics v2:
   - `dependence_metrics_returns.csv`
   - `dependence_metrics_squared_returns.csv`
   - `tail_metrics.csv`
   - `path_risk_metrics.csv`
   - `train_manifest.csv`

### 9.2 Schema v2 Config Contract
New required keys under `schema_version=2`:
- `fit_input_mode`, `return_transform`, `transform_eps`
- `feature_mode`, `feature_rolling_vol_window`
- `eval_tail_quantiles`, `eval_exceedance_thresholds`
- `eval_metrics_version`, `legacy_return_clip`

Backward compatibility:
- `schema_version=1` configs remain runnable.
- If `schema_version` is omitted, v1 validation is applied.

### 9.3 New Experiment Matrix (`baseline_11` ... `baseline_15`)
All five runs use:
- `schema_version=2`
- `fit_input_mode=explicit_windows`
- `match_return_moments=false`
- `input_return_clip_quantiles=null`
- `output_return_clip_quantiles=null`
- train/test split unchanged (`^GSPC`, 2005-2019 train, 2020-2024 test)

Run intents:
- `baseline_11`: control for explicit-window fit with `minmax` + `returns_only`.
- `baseline_12`: isolate effect of `gaussian_cdf` with same hyperparams as 11.
- `baseline_13`: isolate effect of `empirical_cdf` with same hyperparams as 11.
- `baseline_14`: add second feature channel (`returns_plus_abs_return`) with `gaussian_cdf`.
- `baseline_15`: regime-aware variant (`returns_plus_rolling_vol`, higher capacity and epochs) with `empirical_cdf`.

### 9.4 Interpretation Template for Expert Review
For each v2 run, report:
1. Training manifold sanity:
   - `n_windows`, `seq_len`, `n_features`, effective stride from `train_manifest.csv`.
2. Tail behavior:
   - quantile-value errors at 1% and 0.1%,
   - exceedance probability errors at configured thresholds.
3. Dependence behavior:
   - mean absolute lag error for log-returns ACF,
   - mean absolute lag error for squared-log-returns ACF.
4. Path risk:
   - cumulative log-return distribution match,
   - max-drawdown moments and p95 error.
5. Legacy continuity:
   - compare `summary_metrics_test.csv` against `baseline_9` to assess trade-offs.

---

## 10) DDPM v1 Protocol (New Track, TimeGAN Preserved)

### 10.1 Objective
Add a non-adversarial simulator (`DiffusionStock`) based on DDPM/DDIM while keeping TimeGAN fully available.

### 10.2 New Components
1. `src/DeepHedging/HedgingInstruments/diffusion_stock.py`
2. `src/DeepHedging/utils/diffusion_schedule.py`
3. `src/DeepHedging/utils/diffusion_model.py`
4. `src/DeepHedging/utils/diffusion_training.py`
5. `src/DeepHedging/utils/diffusion_sampling.py`
6. `examples/diffusion_simulator_console.py`

### 10.3 DDPM Data Lineage
1. Train/Test CSV cache resolution stays in:
   - `G:/Mi unidad/Tesis2026/TimeGanTraining/market_data_cache`
2. Return features are built from log-returns with configurable `feature_mode`.
3. Explicit windows are constructed in-repo with configured `stride`:
   - tensor shape `(n_windows, n, n_features)`.
4. Feature transforms:
   - channel 0: configured `return_transform` (`minmax`, `gaussian_cdf`, `empirical_cdf`)
   - extra channels: `minmax`
5. DDPM trains on explicit tensor (epsilon prediction MSE).
6. Sampling:
   - `sampler_type=ddpm` (full reverse chain) or `sampler_type=ddim`.
7. Channel 0 inverse-transform -> log-returns -> price paths via cumulative exp.

### 10.4 Diffusion Artifacts
Per run, under `.../<run_name>/`:
1. Model files:
   - `models/diffusion/diffusion_denoiser.keras`
   - `models/diffusion/diffusion_state.npz`
   - `models/diffusion/diffusion_metadata.json`
   - `models/diffusion/diffusion_ema.weights.h5` (if EMA enabled)
2. Scores:
   - `csvs/scores/train_manifest.csv`
   - `csvs/scores/training_loss_history.csv`
   - `csvs/scores/noise_schedule.csv`
   - `csvs/scores/dependence_metrics_returns.csv`
   - `csvs/scores/dependence_metrics_squared_returns.csv`
   - `csvs/scores/tail_metrics.csv`
   - `csvs/scores/path_risk_metrics.csv`
   - legacy-compatible summaries (`summary_metrics_train/test` and aggregated files)

### 10.5 Diffusion Experiment Set
Config files:
1. `gan_training_configs/diffusion_1.json`
2. `gan_training_configs/diffusion_2.json`
3. `gan_training_configs/diffusion_3.json`
4. `gan_training_configs/diffusion_4.json`
5. `gan_training_configs/diffusion_5.json`

Design intent:
1. `diffusion_1`: control (`minmax`, one feature).
2. `diffusion_2`: Gaussian-CDF marginal mapping.
3. `diffusion_3`: empirical-CDF marginal mapping.
4. `diffusion_4`: add abs-return state feature.
5. `diffusion_5`: strongest regime-aware setup (`rolling_vol`, cosine schedule, DDIM).

### 10.6 Reproducibility Commands (Diffusion)
1. `python examples/diffusion_simulator_console.py diffusion_1`
2. `python examples/diffusion_simulator_console.py diffusion_5.json`
