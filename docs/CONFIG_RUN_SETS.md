# Config Run Sets

This document explains the intent of each run subfolder under `configs/runs/`.

## `result1b/euro_gbm/recurrent`

- Goal: controlled GBM experiments for the recurrent agent.
- `train/seen.json`: context is generated and visible to the agent.
- `train/ctxgen.json`: context is generated only to shape paths (`S0` variability), hidden from the agent.
- `train/noctx.json`: no context at all.
- Matching compare configs in `compare/` evaluate against `DeltaHedgingAgent` benchmark.

## `result1b/euro_gbm/wavenet`

- Goal: controlled GBM experiments for WaveNet.
- `train/fixed.json`: fixed sigma.
- `train/pathsigma_uniform.json`: pathwise sigma sampled from uniform range.
- `train/pathsigma_discrete.json`: pathwise sigma sampled from discrete set.
- Matching compare configs in `compare/` evaluate against `DeltaHedgingAgent`.

## `result1b/euro_real/recurrent`

- Goal: train with calibrated setup and test on real windows 2020-2024.
- `train/hist_ctx_mny.json`: recurrent training with historical sigma and context log-moneyness.
- `compare/real_hist_ctx_mny.json`: real-window comparison with rolling historical sigma.

## Runner commands

- Train:
  - `python examples/train_console.py result1b/euro_gbm/recurrent/train/seen`
- Compare:
  - `python examples/compare_console.py result1b/euro_gbm/recurrent/compare/seen`

## `A1/euro_gbm/wavenet`

- Goal: A1 controlled suite requested by user.
- Fixed implied sigma `0.2`, 100 epochs, `ReduceOnPlateau` LR.
- Variants:
  - `train/noctx.json`: no generated history, no visible context.
  - `train/ctxgen.json`: generated history only (hidden from agent).
  - `train/seen.json`: generated history visible to agent.
- Matching compare configs in `compare/` run each variant against `DeltaHedgingAgent`.

## `A2/euro_gbm/wavenet_logk`

- Goal: test different risk measures with the same architecture/data.
- Common setup:
  - WaveNet
  - `log(S)` + extra `log(K)` feature
  - context visible
  - GBM fixed sigma = 0.2
- Variants:
  - `train/cvar95_seen.json`
  - `train/mse_seen.json`
  - `train/mae_seen.json`
- Matching compare configs in `compare/` evaluate each trained model against `DeltaHedgingAgent`.

## `A2/euro_gbm/wavenet_mny`

- Goal: same risk-measure sweep as above, but with `log_moneyness` input.
- Common setup:
  - WaveNet
  - input path transformation: `log_moneyness`
  - context visible
  - GBM fixed sigma = 0.2
- Variants:
  - `train/cvar95_seen.json`
  - `train/mse_seen.json`
  - `train/mae_seen.json`
- Matching compare configs in `compare/` evaluate each trained model against `DeltaHedgingAgent`.
