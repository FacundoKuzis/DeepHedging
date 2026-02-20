import numpy as np
import pandas as pd

from DeepHedging.utils.market_data import load_prices_csv



def build_historical_windows_from_csv(
    csv_path: str,
    n_hedging_steps: int,
    start_date: str,
    end_date: str,
    price_col: str = "Close",
    stride: int = 1,
    max_windows: int | None = None,
):
    """
    Build fixed-length rolling windows from historical prices.

    Returns:
        windows_2d: np.ndarray shape (n_windows, n_hedging_steps+1)
        metadata: pd.DataFrame with window-level information
    """
    if int(n_hedging_steps) <= 0:
        raise ValueError("n_hedging_steps must be > 0.")
    if int(stride) <= 0:
        raise ValueError("stride must be > 0.")
    if max_windows is not None and int(max_windows) <= 0:
        raise ValueError("max_windows must be > 0 when provided.")

    df = load_prices_csv(csv_path, price_col=price_col).copy()
    if "Date" not in df.columns:
        raise ValueError(f"CSV does not contain Date column: {csv_path}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No market rows in requested range {start_date}..{end_date} from {csv_path}."
        )

    prices = df[price_col].to_numpy(dtype=np.float64)
    dates = df["Date"].to_numpy()
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0.0):
        raise ValueError("Historical prices must be finite and strictly positive.")

    window_len = int(n_hedging_steps) + 1
    n_total = int(prices.shape[0])
    max_start = n_total - window_len
    if max_start < 0:
        raise ValueError(
            f"Not enough rows ({n_total}) to build windows of length {window_len}."
        )

    windows = []
    rows = []
    wid = 0
    for start_idx in range(0, max_start + 1, int(stride)):
        end_idx = start_idx + window_len - 1
        w = prices[start_idx : end_idx + 1]
        windows.append(w)
        rows.append(
            {
                "window_id": int(wid),
                "row_start_idx": int(start_idx),
                "row_end_idx": int(end_idx),
                "start_date": pd.Timestamp(dates[start_idx]),
                "end_date": pd.Timestamp(dates[end_idx]),
                "s0": float(w[0]),
                "terminal_price": float(w[-1]),
                "terminal_simple_return": float(w[-1] / w[0] - 1.0),
            }
        )
        wid += 1
        if max_windows is not None and wid >= int(max_windows):
            break

    if not windows:
        raise ValueError("No windows built from historical data.")

    windows_2d = np.asarray(windows, dtype=np.float32)
    metadata = pd.DataFrame(rows)
    return windows_2d, metadata



def to_environment_paths(windows_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows_2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"windows_2d must be rank-2, got shape={arr.shape}.")
    return np.expand_dims(arr, axis=-1).astype(np.float32)
