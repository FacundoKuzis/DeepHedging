# Unified Configs

This folder centralizes train/compare configs with JSON inheritance.

## Structure

- `bases/`: reusable templates with common parameters.
- `runs/`: executable experiments, grouped by study set.

Current naming direction:

- Keep file names short.
- Encode context in folders.
- Example: `runs/result1b/euro_gbm/recurrent/train/seen.json`
  instead of a long monolithic filename.

## Inheritance

A config can inherit from one or many parents with `extends`.

Example:

```json
{
  "extends": "thesis_result1b_configs/train/euro_train_gbm_fixed_recurrent_log_moneyness_increased_vol_ctx_42.json",
  "run_name": "euro_gbm_rec_seen",
  "model_name": "euro_gbm_rec_seen"
}
```

Resolution order:

1. Parent(s) load first.
2. Child keys override parent keys.
3. Nested dictionaries are deep-merged.

## Unified runners

- Train:
  - `python examples/train_console.py <config_ref>`
- Compare:
  - `python examples/compare_console.py <config_ref>`

`<config_ref>` can be:

- Relative path from `configs/`
- Relative path from repo root
- Filename (if unique under `configs/runs/**/{train|compare}`)

## Suggested short refs

Train recurrent GBM with context visible:

- `python examples/train_console.py result1b/euro_gbm/recurrent/train/seen`

Compare that run:

- `python examples/compare_console.py result1b/euro_gbm/recurrent/compare/seen`

Train recurrent GBM with context generated-only:

- `python examples/train_console.py result1b/euro_gbm/recurrent/train/ctxgen`

Compare that run:

- `python examples/compare_console.py result1b/euro_gbm/recurrent/compare/ctxgen`

## A1 set (requested)

WaveNet sigma 0.2, no context:

- `python examples/train_console.py A1/euro_gbm/wavenet/train/noctx`
- `python examples/compare_console.py A1/euro_gbm/wavenet/compare/noctx`

WaveNet sigma 0.2, context generated-only (hidden):

- `python examples/train_console.py A1/euro_gbm/wavenet/train/ctxgen`
- `python examples/compare_console.py A1/euro_gbm/wavenet/compare/ctxgen`

WaveNet sigma 0.2, context generated+visible:

- `python examples/train_console.py A1/euro_gbm/wavenet/train/seen`
- `python examples/compare_console.py A1/euro_gbm/wavenet/compare/seen`

WaveNet sigma 0.2, input `log(S)` + additional `log(K)` feature:

- no context:
  - `python examples/train_console.py A1/euro_gbm/wavenet_logk/train/noctx`
  - `python examples/compare_console.py A1/euro_gbm/wavenet_logk/compare/noctx`
- context generated-only (hidden):
  - `python examples/train_console.py A1/euro_gbm/wavenet_logk/train/ctxgen`
  - `python examples/compare_console.py A1/euro_gbm/wavenet_logk/compare/ctxgen`
- context generated+visible:
  - `python examples/train_console.py A1/euro_gbm/wavenet_logk/train/seen`
  - `python examples/compare_console.py A1/euro_gbm/wavenet_logk/compare/seen`

## A2 set (risk objective sweep)

Same setup as `A1/euro_gbm/wavenet_logk/.../seen`, changing only training objective:

- CVaR95:
  - `python examples/train_console.py A2/euro_gbm/wavenet_logk/train/cvar95_seen`
  - `python examples/compare_console.py A2/euro_gbm/wavenet_logk/compare/cvar95_seen`
  - (also available as CVaR50: `.../cvar50_seen` in matching folders)
- MSE:
  - `python examples/train_console.py A2/euro_gbm/wavenet_logk/train/mse_seen`
  - `python examples/compare_console.py A2/euro_gbm/wavenet_logk/compare/mse_seen`
- MAE:
  - `python examples/train_console.py A2/euro_gbm/wavenet_logk/train/mae_seen`
  - `python examples/compare_console.py A2/euro_gbm/wavenet_logk/compare/mae_seen`

Same sweep also available for `log_moneyness` input:

- CVaR95:
  - `python examples/train_console.py A2/euro_gbm/wavenet_mny/train/cvar95_seen`
  - `python examples/compare_console.py A2/euro_gbm/wavenet_mny/compare/cvar95_seen`
  - (also available as CVaR50: `.../cvar50_seen`, `.../cvar50_noctx`, `.../cvar50_ctxgen`)
- MSE:
  - `python examples/train_console.py A2/euro_gbm/wavenet_mny/train/mse_seen`
  - `python examples/compare_console.py A2/euro_gbm/wavenet_mny/compare/mse_seen`
- MAE:
  - `python examples/train_console.py A2/euro_gbm/wavenet_mny/train/mae_seen`
  - `python examples/compare_console.py A2/euro_gbm/wavenet_mny/compare/mae_seen`

Also available for `wavenet_mny` with context variants:

- no context:
  - `.../train/cvar95_noctx`, `.../train/mse_noctx`, `.../train/mae_noctx`
  - `.../compare/cvar95_noctx`, `.../compare/mse_noctx`, `.../compare/mae_noctx`
- generated-only context (hidden):
  - `.../train/cvar95_ctxgen`, `.../train/mse_ctxgen`, `.../train/mae_ctxgen`
  - `.../compare/cvar95_ctxgen`, `.../compare/mse_ctxgen`, `.../compare/mae_ctxgen`

## A3 set (random volatility per path)

WaveNet + log-moneyness, objective CVaR50, no transaction costs.

- Uniform narrow:
  - `python examples/train_console.py A3/euro_gbm/wavenet_mny_randvol/train/uniform_narrow`
  - `python examples/compare_console.py A3/euro_gbm/wavenet_mny_randvol/compare/uniform_narrow`
- Uniform wide:
  - `python examples/train_console.py A3/euro_gbm/wavenet_mny_randvol/train/uniform_wide`
  - `python examples/compare_console.py A3/euro_gbm/wavenet_mny_randvol/compare/uniform_wide`
- Discrete regimes:
  - `python examples/train_console.py A3/euro_gbm/wavenet_mny_randvol/train/discrete_regimes`
  - `python examples/compare_console.py A3/euro_gbm/wavenet_mny_randvol/compare/discrete_regimes`

## Note

Legacy folders (`thesis_result1_configs`, `thesis_result1b_configs`) remain usable.
This folder is the ordered, scalable replacement.
