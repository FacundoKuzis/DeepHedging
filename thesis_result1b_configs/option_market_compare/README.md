# Option Market Compare Configs

Estos JSON se usan con:

python examples/thesis_result1b_option_market_compare_console.py <nombre_config>

Columnas mínimas esperadas en `option_quotes_csv`:
- quote_date
- expiration
- strike
- option_type
- bid
- ask
- last

Regla de matching por ventana:
1. Se busca fecha de cotización dentro de `max_quote_lag_days` alrededor de `window_start_date`.
2. Se elige vencimiento más cercano al objetivo de días calendario equivalente a `n` días de trading.
3. Se elige strike más cercano al precio spot inicial de la ventana.

Precios comparados:
- `bs_price_252`: con T = n / trading_days_per_year
- `bs_price_calendar`: con T = days_to_exp / calendar_days_per_year
- `market_option_price`: mid (bid+ask)/2 cuando existe; si no, fallback a last/bid/ask
