import os
import time

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


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {str(col).strip().lower(): col for col in df.columns}
    rename = {}
    expected = {
        "date": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }
    for key, target in expected.items():
        if key in col_map:
            rename[col_map[key]] = target
    if rename:
        df = df.rename(columns=rename)
    return df


def _finalize_market_df(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        else:
            return pd.DataFrame()

    df = _normalize_ohlcv_columns(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        return pd.DataFrame()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()
    if df.empty:
        return pd.DataFrame()

    if "Close" in df.columns and "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df


def _download_with_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    retries: int = 3,
    retry_sleep_sec: float = 2.0,
) -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "download_ohlcv_to_csv requires 'yfinance'. Install it with `pip install yfinance`."
        ) from exc

    last_reason = ""
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
                actions=False,
                threads=False,
                group_by="column",
            )
            if data is not None and not data.empty:
                data = data.reset_index()
                data = _flatten_yfinance_columns(data, ticker=ticker)
                data = _finalize_market_df(data, start_date=start_date, end_date=end_date)
                if not data.empty:
                    return data, "yfinance.download"
        except Exception as exc:
            last_reason = f"yf.download attempt {attempt} failed: {exc!r}"

        try:
            data = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )
            if data is not None and not data.empty:
                data = data.reset_index()
                data = _normalize_ohlcv_columns(data)
                data = _finalize_market_df(data, start_date=start_date, end_date=end_date)
                if not data.empty:
                    return data, "yfinance.Ticker.history"
        except Exception as exc:
            if last_reason:
                last_reason = f"{last_reason}; yf.Ticker.history attempt {attempt} failed: {exc!r}"
            else:
                last_reason = f"yf.Ticker.history attempt {attempt} failed: {exc!r}"

        if attempt < retries:
            time.sleep(retry_sleep_sec * attempt)

    return pd.DataFrame(), (last_reason or "yfinance returned empty data.")


def _download_from_stooq(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> tuple[pd.DataFrame, str]:
    interval_map = {"1d": "d", "1wk": "w", "1mo": "m"}
    if interval not in interval_map:
        return pd.DataFrame(), f"stooq fallback not available for interval '{interval}'"

    ticker_key = ticker.strip().lower()
    index_aliases = {
        "^gspc": ["^spx", "spx.us", "^gspc"],
        "^spx": ["^spx", "spx.us"],
        "spx": ["spx.us", "^spx"],
    }
    symbol_candidates = index_aliases.get(ticker_key, [ticker_key])
    if "." not in ticker_key and not ticker_key.startswith("^") and ticker_key not in index_aliases:
        symbol_candidates.insert(0, f"{ticker_key}.us")

    last_reason = ""
    for symbol in symbol_candidates:
        url = f"https://stooq.com/q/d/l/?s={symbol}&i={interval_map[interval]}"
        try:
            data = pd.read_csv(url)
            data = _normalize_ohlcv_columns(data)
            data = _finalize_market_df(data, start_date=start_date, end_date=end_date)
            if data is not None and not data.empty:
                return data, f"stooq:{symbol}"
            last_reason = f"stooq:{symbol} returned empty data."
        except Exception as exc:
            last_reason = f"stooq:{symbol} failed: {exc!r}"

    return pd.DataFrame(), (last_reason or "stooq fallback failed.")


def download_ohlcv_to_csv(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    output_csv: str,
) -> str:
    if not ticker:
        raise ValueError("ticker must be a non-empty string.")

    data, source_used = _download_with_yfinance(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    if data is None or data.empty:
        fallback_data, fallback_source = _download_from_stooq(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        if fallback_data is None or fallback_data.empty:
            raise ValueError(
                "No data downloaded. Primary source (Yahoo/yfinance) failed and fallback (Stooq) also failed. "
                f"ticker='{ticker}', start='{start_date}', end='{end_date}', interval='{interval}'. "
                f"Primary detail: {source_used}. Fallback detail: {fallback_source}."
            )
        data = fallback_data
        source_used = fallback_source

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    data.to_csv(output_csv, index=False)

    actual_start = pd.to_datetime(data["Date"], errors="coerce").min()
    actual_end = pd.to_datetime(data["Date"], errors="coerce").max()
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    if actual_start > requested_start or actual_end < requested_end:
        print(
            f"[market_data][warning] Requested range {start_date}..{end_date} "
            f"is not fully covered by source='{source_used}'. "
            f"Available range: {actual_start.date()}..{actual_end.date()}."
        )

    print(
        f"[market_data] Saved {len(data)} rows for ticker='{ticker}' "
        f"from source='{source_used}' to '{output_csv}'."
    )
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
