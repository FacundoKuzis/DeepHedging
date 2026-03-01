# DeepHedging Repository Capabilities Catalog (Current Code State)

## 1. Scope of this document
This document is a code-grounded inventory of what the repository can do today, oriented to an expert committee deciding thesis experiment scenarios.

It covers:
- what can be trained,
- what can be compared against what,
- supported market/path generators,
- supported risk objectives and evaluation modes,
- operational constraints that materially affect experiment design.

All items below are based on current source code in:
- `examples/train_console.py`
- `examples/compare_console.py`
- `examples/thesis_result1b_train_console.py`
- `examples/thesis_result1b_compare_console.py`
- `examples/thesis_result1_common.py`
- `src/DeepHedging/HedgingInstruments/stock.py`
- `src/DeepHedging/Agents/local_risk_minimization_agent.py`
- `src/DeepHedging/utils/lrm_continuation.py`
- `src/DeepHedging/utils/lrm_providers.py`

## 2. Runner architecture
Unified runners:
- `python examples/train_console.py <config_ref>`
- `python examples/compare_console.py <config_ref>`

Supported pipelines:
- `result1`
- `result1b`
- `result1b_option_market` (compare-oriented specialized flow)

Config resolution:
- JSON inheritance via `extends` (single string or list), deep-merge, then strict validation.
- Unified runners apply relaxed defaults for many optional fields before dispatching to pipeline-specific validators.

## 3. Config organization and output organization
Config tree:
- `configs/bases/...` for reusable bases.
- `configs/runs/<SET>/...` for concrete experiments (A1, A2, ..., B1, ...).

Output root:
- Main organized root is `G:\Mi unidad\Tesis2026\Models\Organized` (via `THESIS_MODELS_ROOT`).
- Run artifacts are saved mirroring config relative path structure.

Typical run artifacts:
- config snapshot,
- trained model and optimizer snapshots,
- tables (`point_metrics`, `empirical_risk_metrics`, bootstrap),
- plots,
- logs,
- compare action caches.

## 4. Trainable agents
Trainable agent names accepted by result1b:
- `SimpleAgent`
- `RecurrentAgent`
- `LSTMAgent`
- `GRUAgent`
- `WaveNetAgent`

Common train-time features:
- path transform input mode: `none`, `log`, `log_moneyness`,
- optional extra input feature `include_log_strike_feature`,
- sequence output mode support (`trade` or `position` where applicable),
- position activation selection,
- optional history context features,
- optional Conv1D history encoder attached to trainable agents.

WaveNet-specific configurable blocks:
- legacy knobs: `wavenet_num_filters`, `wavenet_num_residual_blocks`,
- structured blocks: `wavenet_block_configs`,
- `wavenet_activation`,
- `wavenet_use_skip_connections`,
- `wavenet_output_hidden_filters`.

## 5. Benchmark / non-trainable agents available in compare
Non-trainable agents exposed by current code:
- `DeltaHedgingAgent`
- `GeometricAsianDeltaHedgingAgent`
- `GeometricAsianDeltaHedgingAgent2`
- `GeometricAsianNumericalDeltaHedgingAgent`
- `QuantlibAsianGeometricAgent`
- `ArithmeticAsianMonteCarloAgent`
- `ArithmeticAsianControlVariateAgent`
- `MonteCarloAgent`
- `LocalRiskMinimizationAgent`

Compare supports:
- one main benchmark (`benchmark_agent_name`),
- one or more trained agents (`trained_agents`),
- optional extra benchmark list (`benchmark_agents_to_compare`) with per-agent config overrides.

## 6. LRM benchmark stack (implemented)
`LocalRiskMinimizationAgent` uses one-step LRM target:
- `h_t = Cov(C_{t+1}, dS_{t+1}) / Var(dS_{t+1})`.

Pluggable continuation providers:
- `bs_closed_form` (European only),
- `monte_carlo` (generic),
- `lsm` / `lsmc` (generic regression continuation),
- `asian_monte_carlo`,
- `asian_lsmc`.

Implemented provider controls include:
- outer paths,
- inner MC paths and chunking,
- parallelization (`thread` or `process`, workers, chunk size),
- antithetic usage,
- seed modes,
- LSM controls (train paths, ridge alpha, polynomial degree, feature set),
- LSM cache options (`use_cache`, cache dir/key, force rebuild),
- verbosity and progress logging cadence.

## 7. Contingent claims currently supported
Claims exposed in result1/result1b builders:
- `EuropeanCall`
- `EuropeanPut`
- `AsianGeometricCall`
- `AsianGeometricPut`
- `AsianArithmeticCall`
- `AsianArithmeticPut`

Claim config supports:
- strike,
- underlying index,
- optional fixing indices.

## 8. Path generators / market worlds supported
Instrument models currently validated for result1b:
- `gbm`
- `garch`
- `hmm_garch`
- `student_t`

Also exported in package (not the main result1b train/compare validator path by default):
- `HestonStock`
- `TimeGANStock`
- `DiffusionStock`

### 8.1 GBM
Supports per-path sigma sampling:
- fixed,
- uniform range,
- discrete values (optional discrete probabilities).

### 8.2 Student-t stock
Log-return innovations use standardized Student-t shocks (unit variance normalization).

Supports per-path:
- sigma mode: fixed/uniform/discrete,
- degrees of freedom mode: fixed/uniform/discrete.

### 8.3 GARCH stock
Supports:
- GARCH(1,1)-style conditional variance with optional leverage term,
- optional Student-t innovations,
- per-path randomization for:
  - `r`,
  - long-run sigma,
  - `alpha`,
  - `beta`,
  - leverage,
  - Student-t df.

Stationarity checks enforced:
- `alpha + beta < 1`,
- `alpha + beta + 2*leverage < 1`.

It stores both:
- ex-ante sampled long-run sigma (`_last_sampled_sigmas`),
- realized effective sigma from simulated variance path (`_last_realized_sigmas`).

### 8.4 HMM + GARCH stock
Generic N-state HMM on top of GARCH (no hardcoded 2/3-state logic required).

HMM parameter modes:
- `fixed`:
  - transition matrix,
  - initial distribution,
  - per-state volatility multipliers.
- `uniform_random`:
  - random transition matrices per path (row-normalized),
  - random initial distributions per path (normalized),
  - random state multipliers per path,
  - optional sorting of multipliers.

Also combined with per-path randomization of GARCH and Student-t parameters as above.

## 9. Risk objectives in training
Result1b training builder currently supports:
- `MSE`
- `MAE`
- `CVaR` (including shorthand forms like `CVaR95`, `CVaR50`)

Risk objective is configured by:
- `risk_measure_name`,
- `cvar_alpha` when generic `CVaR` is used.

## 10. Learning-rate and stopping controls
Learning-rate strategies:
- `constant`
- `exponential_decay`
- `reduce_on_plateau`

Reduce-on-plateau controls:
- factor, patience, min delta, cooldown, min lr.

Early stopping controls:
- enabled flag,
- patience,
- min delta.

Checkpointing (implemented):
- periodic latest checkpoint,
- best checkpoint by metric (`auto`, `val_loss`, `train_loss`),
- optional optimizer checkpoint,
- resume from latest checkpoint when available.

Validation path in `Environment.train`:
- now executed in mini-batches to reduce OOM risk on large models/context.

## 11. Context features and context usage modes
Context controls:
- `use_price_history_context`,
- `context_length`,
- `context_feature_mode` in `{log_returns, log_moneyness}`,
- `context_pre_ttm_mode` in `{calculated, zero}`,
- `context_for_path_generation_only`.

Conceptual modes:
- no context,
- context generated only (used to generate path history, hidden from agent),
- context seen by agent.

Conv1D history encoder:
- `history_conv1d_enabled`,
- `history_conv1d_layers` (list of dicts, required when enabled),
- `history_conv1d_pooling` (`global_max` or `global_avg`).

Important behavior:
- if Conv1D is enabled on sequence agents, context is routed as feature-encoded history (not as temporal prefix mode).

## 12. Compare-time pricing and fairness mechanics
Price computation controls accepted:
- `pricing_method`: `fixed` or `individual`,
- `price_computation_mode`: `pathwise_if_available` or `scalar`.

Effective compare behavior in current result1b compare implementation:
- comparison forces effective `pricing_method='fixed'`,
- all agents are evaluated against the first-agent pathwise price vector.

This is explicitly logged in compare when another pricing mode is requested.

## 13. Volatility/rate calibration and compare-time sigma/r logic
Calibration sources:
- `sigma_source`: `historical` or `implied`,
- implied source supports `vix`, `fixed`, and option-market pathway in result1b compare,
- `risk_free_source`: `irx` or `fixed`.

Sigma mapping modes in compare:
- `train_average`,
- `per_window_start` (for implied case where supported),
- `rolling_pre_window` (historical-only).

Risk-free mapping modes:
- `train_average`,
- `per_window_start`.

## 14. Delta benchmark sigma estimators in compare
`benchmark_delta_sigma_mode` supports:
- `none`,
- `garch_context_static`,
- `garch_context_stepwise`,
- `hmm_garch_student_context_stepwise`.

Additional benchmark estimation options:
- GARCH fit from context windows,
- Student-t fit over standardized residuals (`static` or `stepwise`),
- HMM state/tail adjustment controls for the HMM-GARCH-Student mode.

## 15. Evaluation modes
Compare test data modes:
- `simulated`,
- `historical_windows`.

Historical mode:
- builds rolling windows from market CSV/downloaded data,
- supports stride and max window caps,
- optional option-market implied-vol mapping workflow in compare.

Simulated mode:
- uses selected instrument model and parameter sampling config to generate paths.

## 16. Bootstrap and metrics/reporting
Bootstrap:
- can be enabled/disabled,
- methods: `iid` and `moving_block`,
- configurable confidence and batch parameters.

Output tables generated by compare include:
- point metrics,
- empirical risk metrics,
- bootstrap tables (when enabled).

Consolidation/report tooling:
- `examples/thesis_result1b_report_console.py` consolidates multiple compare runs and builds ranking files.

## 17. Path visualization tooling
Script:
- `examples/thesis_result1b_plot_paths_console.py`

Outputs:
- sampled paths in levels `S`,
- sampled paths in `ln(S/K)`,
- sampled paths CSV,
- sampled sigma histogram and sigma CSV when available from instrument sampler.

Works with unified config references and inherits compare defaults.

## 18. Option-market comparison tooling
Specialized runner:
- `examples/thesis_result1b_option_market_compare_console.py`

It compares model-theoretical prices vs historical option quotes window-by-window and saves:
- window-level tables,
- summary tables,
- market-vs-BS plots,
- error time-series plots.

## 19. Important operational constraints for experiment design
1. For result1b compare, effective pricing is fixed to first-agent pathwise price for all agents.
2. Result1b train/compare validators currently accept `instrument_model` in `{gbm, garch, hmm_garch, student_t}`.
3. Stationarity constraints are enforced for GARCH/HMM-GARCH sampled parameters.
4. Conv1D history requires context enabled and not context-generation-only.
5. Some legacy scripts still point to legacy result folders; unified runners and organized config tree should be preferred for new thesis runs.

## 20. Recommended committee-facing experiment menu (based on available features)
You can safely propose scenarios along these axes:
- claim type: European vs Asian geometric vs Asian arithmetic,
- world generator: GBM vs GARCH vs HMM+GARCH vs Student-t,
- parameter heterogeneity: fixed vs per-path sampled distributions,
- context regime: no context vs generated-only vs seen + Conv1D encoder,
- risk objective: MSE vs MAE vs CVaR variants,
- benchmark family: BS-delta variants, Asian benchmarks, LRM with MC/LSM/Asian providers,
- evaluation domain: simulated vs historical windows,
- sigma/r estimation policy for benchmark.

This space is already implementable with current code and config schema.
