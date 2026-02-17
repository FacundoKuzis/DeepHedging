import os

import pandas as pd


def _flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for col in df.columns.to_flat_index():
            parts = [str(part) for part in col if str(part) != ""]
            flattened.append("_".join(parts))
        df.columns = flattened

    ticker_suffix = f"_{ticker}"
    renamed = {}
    for col in df.columns:
        if col.endswith(ticker_suffix):
            renamed[col] = col[: -len(ticker_suffix)]
    if renamed:
        df = df.rename(columns=renamed)
    return df


def download_ohlcv_to_csv(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    output_csv: str,
) -> str:
    if not ticker:
        raise ValueError("ticker must be a non-empty string.")

    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - dependency import failure
        raise ImportError(
            "download_ohlcv_to_csv requires 'yfinance'. Install it with `pip install yfinance`."
        ) from exc

    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"No data downloaded for ticker='{ticker}', start='{start_date}', end='{end_date}', interval='{interval}'."
        )

    data = data.reset_index()
    data = _flatten_yfinance_columns(data, ticker=ticker)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    data.to_csv(output_csv, index=False)
    return output_csv


def load_prices_csv(csv_path: str, price_col: str = "Close") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    col_map = {col.lower(): col for col in df.columns}
    if price_col not in df.columns:
        normalized_key = price_col.lower()
        if normalized_key in col_map:
            price_col = col_map[normalized_key]
        else:
            raise ValueError(
                f"Price column '{price_col}' not found in CSV. Available columns: {list(df.columns)}"
            )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["Datetime"], errors="coerce")

    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"CSV has no valid rows after dropping NaNs in '{price_col}': {csv_path}")

    return df
