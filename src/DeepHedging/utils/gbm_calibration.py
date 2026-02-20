import os
import warnings
from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np
import pandas as pd

from DeepHedging.utils.market_data import download_ohlcv_to_csv, load_prices_csv


VIX_TICKER = "^VIX"
IRX_TICKER = "^IRX"


@dataclass
class CalibratedGBMParameters:
    sigma_train: float
    r_train: float
    mu_train: float
    sigma_source: str
    risk_free_source: str
    train_rows: int
    close_csv_path: str
    implied_csv_path: str | None = None
    risk_free_csv_path: str | None = None



def _ensure_csv(
    cache_dir: str,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    download_if_missing: bool,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{ticker}_{start_date}_{end_date}_{interval}.csv"
    csv_path = os.path.join(cache_dir, filename)
    if os.path.isfile(csv_path):
        return csv_path
    if not download_if_missing:
        raise FileNotFoundError(
            f"Market cache CSV not found and download_if_missing=False: {csv_path}"
        )
    download_ohlcv_to_csv(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        output_csv=csv_path,
    )
    return csv_path



def _load_close_series(csv_path: str, price_col: str = "Close") -> pd.DataFrame:
    df = load_prices_csv(csv_path, price_col=price_col).copy()
    if "Date" not in df.columns:
        raise ValueError(f"CSV does not contain Date column: {csv_path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)
    return df[["Date", price_col]].copy()



def _annualized_hist_vol(close_prices: np.ndarray, trading_days_per_year: int) -> float:
    prices = np.asarray(close_prices, dtype=np.float64).reshape(-1)
    prices = prices[np.isfinite(prices)]
    if prices.size < 3:
        raise ValueError("Need at least 3 close prices to estimate historical volatility.")
    if np.any(prices <= 0.0):
        raise ValueError("Close prices must be strictly positive for log-return volatility.")
    log_returns = np.diff(np.log(prices))
    if log_returns.size < 2:
        raise ValueError("Not enough log returns to estimate historical volatility.")
    sigma_daily = np.std(log_returns, ddof=1)
    return float(sigma_daily * np.sqrt(float(trading_days_per_year)))



def _average_percent_to_decimal(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot calibrate from empty series.")
    return float(np.mean(arr) / 100.0)



def _median_percent_to_decimal(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot calibrate from empty series.")
    return float(np.median(arr) / 100.0)



def calibrate_gbm_from_market_data(
    market_cache_dir: str,
    ticker: str,
    train_start_date: str,
    train_end_date: str,
    interval: str,
    price_col: str,
    trading_days_per_year: int,
    sigma_source: str,
    implied_vol_source: str,
    implied_vol_stat: str,
    fixed_implied_vol: float | None,
    risk_free_source: str,
    fixed_risk_free: float | None,
    download_if_missing: bool,
) -> CalibratedGBMParameters:
    sigma_source = str(sigma_source).strip().lower()
    implied_vol_source = str(implied_vol_source).strip().lower()
    implied_vol_stat = str(implied_vol_stat).strip().lower()
    risk_free_source = str(risk_free_source).strip().lower()

    if sigma_source not in {"historical", "implied"}:
        raise ValueError("sigma_source must be 'historical' or 'implied'.")
    if implied_vol_source not in {"vix", "fixed", "option_market"}:
        raise ValueError("implied_vol_source must be 'vix', 'fixed' or 'option_market'.")
    if implied_vol_stat not in {"mean", "median"}:
        raise ValueError("implied_vol_stat must be 'mean' or 'median'.")
    if risk_free_source not in {"irx", "fixed"}:
        raise ValueError("risk_free_source must be 'irx' or 'fixed'.")

    close_csv = _ensure_csv(
        cache_dir=market_cache_dir,
        ticker=ticker,
        start_date=train_start_date,
        end_date=train_end_date,
        interval=interval,
        download_if_missing=download_if_missing,
    )
    close_df = _load_close_series(close_csv, price_col=price_col)

    implied_csv = None
    risk_free_csv = None

    def _load_with_fallback_candidates(candidates: list[str], label: str) -> tuple[pd.DataFrame, str, str]:
        last_exc = None
        for candidate in candidates:
            try:
                csv_candidate = _ensure_csv(
                    cache_dir=market_cache_dir,
                    ticker=candidate,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    interval=interval,
                    download_if_missing=download_if_missing,
                )
                df_candidate = _load_close_series(csv_candidate, price_col="Close")
                if not df_candidate.empty:
                    return df_candidate, csv_candidate, candidate
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise ValueError(f"Unable to load {label} from candidates={candidates}. Last error: {last_exc}") from last_exc
        raise ValueError(f"Unable to load {label} from candidates={candidates}.")

    if sigma_source == "historical":
        sigma_train = _annualized_hist_vol(
            close_df[price_col].to_numpy(dtype=np.float64),
            trading_days_per_year=int(trading_days_per_year),
        )
    else:
        if implied_vol_source == "fixed":
            if fixed_implied_vol is None:
                raise ValueError("fixed_implied_vol must be provided when implied_vol_source='fixed'.")
            sigma_train = float(fixed_implied_vol)
        elif implied_vol_source == "vix":
            implied_df, implied_csv, implied_ticker = _load_with_fallback_candidates(
                candidates=[VIX_TICKER, "VIX"],
                label="implied volatility",
            )
            if implied_vol_stat == "median":
                sigma_train = _median_percent_to_decimal(implied_df["Close"].to_numpy(dtype=np.float64))
            else:
                sigma_train = _average_percent_to_decimal(implied_df["Close"].to_numpy(dtype=np.float64))
        else:
            # option_market implies pathwise IV mapping in compare/eval stage.
            # For train-level sigma we require an explicit fallback or use historical sigma.
            if fixed_implied_vol is not None:
                sigma_train = float(fixed_implied_vol)
            else:
                warnings.warn(
                    "implied_vol_source='option_market' without fixed_implied_vol: "
                    "falling back to historical sigma on train window for sigma_train.",
                    RuntimeWarning,
                )
                sigma_train = _annualized_hist_vol(
                    close_df[price_col].to_numpy(dtype=np.float64),
                    trading_days_per_year=int(trading_days_per_year),
                )

    if sigma_train <= 0.0:
        raise ValueError(f"Calibrated sigma must be > 0. Got {sigma_train}")

    if risk_free_source == "fixed":
        if fixed_risk_free is None:
            raise ValueError("fixed_risk_free must be provided when risk_free_source='fixed'.")
        r_train = float(fixed_risk_free)
    else:
        risk_candidates = [
            (IRX_TICKER, 0.01),  # ^IRX in percent points
            ("IRX", 0.01),       # alt Yahoo symbol
            ("^FVX", 0.001),     # often 10x yield percent
            ("^TNX", 0.001),     # often 10x yield percent
        ]
        last_exc = None
        risk_free_df = None
        risk_scale = None
        risk_ticker = None
        for ticker_candidate, scale_candidate in risk_candidates:
            try:
                csv_candidate = _ensure_csv(
                    cache_dir=market_cache_dir,
                    ticker=ticker_candidate,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    interval=interval,
                    download_if_missing=download_if_missing,
                )
                df_candidate = _load_close_series(csv_candidate, price_col="Close")
                if not df_candidate.empty:
                    risk_free_df = df_candidate
                    risk_free_csv = csv_candidate
                    risk_scale = float(scale_candidate)
                    risk_ticker = ticker_candidate
                    break
            except Exception as exc:
                last_exc = exc
                continue
        if risk_free_df is None:
            if fixed_risk_free is not None:
                warnings.warn(
                    "Risk-free source 'irx' unavailable; falling back to fixed_risk_free value.",
                    RuntimeWarning,
                )
                r_train = float(fixed_risk_free)
            else:
                warnings.warn(
                    "Risk-free source 'irx' unavailable and fixed_risk_free not provided; "
                    "falling back to r_train=0.0.",
                    RuntimeWarning,
                )
                r_train = 0.0
        else:
            r_train = float(np.mean(risk_free_df["Close"].to_numpy(dtype=np.float64)) * risk_scale)

    mu_train = float(r_train)

    return CalibratedGBMParameters(
        sigma_train=float(sigma_train),
        r_train=float(r_train),
        mu_train=float(mu_train),
        sigma_source=sigma_source,
        risk_free_source=risk_free_source,
        train_rows=int(close_df.shape[0]),
        close_csv_path=close_csv,
        implied_csv_path=implied_csv,
        risk_free_csv_path=risk_free_csv,
    )



def map_series_to_window_start(
    series_df: pd.DataFrame,
    window_start_dates: Iterable[pd.Timestamp],
    value_col: str,
    scale: float = 1.0,
    default_value: float | None = None,
) -> np.ndarray:
    """
    Map time-series values to each window start date using backward asof lookup.
    """
    if value_col not in series_df.columns:
        raise ValueError(f"value_col '{value_col}' not found in series_df columns.")

    src = series_df.copy()
    src["Date"] = pd.to_datetime(src["Date"], errors="coerce")
    src[value_col] = pd.to_numeric(src[value_col], errors="coerce")
    src = src.dropna(subset=["Date", value_col]).sort_values("Date").reset_index(drop=True)
    if src.empty:
        if default_value is None:
            raise ValueError("Cannot map from empty source series.")
        starts = list(window_start_dates)
        return np.full((len(starts),), float(default_value), dtype=np.float64)

    q = pd.DataFrame({"Date": pd.to_datetime(list(window_start_dates), errors="coerce")})
    merged = pd.merge_asof(
        q.sort_values("Date"),
        src[["Date", value_col]].sort_values("Date"),
        on="Date",
        direction="backward",
    )
    if merged[value_col].isna().any():
        merged[value_col] = merged[value_col].ffill().bfill()
    if merged[value_col].isna().any():
        if default_value is None:
            raise ValueError("Unable to map values for all window start dates.")
        merged[value_col] = merged[value_col].fillna(float(default_value))

    values = merged[value_col].to_numpy(dtype=np.float64) * float(scale)
    return values



def load_external_series(
    market_cache_dir: str,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    download_if_missing: bool,
    price_col: str = "Close",
) -> pd.DataFrame:
    csv_path = _ensure_csv(
        cache_dir=market_cache_dir,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        download_if_missing=download_if_missing,
    )
    return _load_close_series(csv_path, price_col=price_col)


def map_historical_sigma_to_window_start(
    close_df: pd.DataFrame,
    window_start_dates: Iterable[pd.Timestamp],
    price_col: str,
    window_days: int,
    trading_days_per_year: int,
    default_value: float | None = None,
) -> np.ndarray:
    """
    Map annualized historical volatility to each window start date using only
    observations strictly prior to each start date.
    """
    if int(window_days) < 2:
        raise ValueError("window_days must be >= 2.")
    if int(trading_days_per_year) <= 0:
        raise ValueError("trading_days_per_year must be > 0.")
    if price_col not in close_df.columns:
        raise ValueError(f"price_col '{price_col}' not found in close_df.")

    src = close_df.copy()
    src["Date"] = pd.to_datetime(src["Date"], errors="coerce")
    src[price_col] = pd.to_numeric(src[price_col], errors="coerce")
    src = src.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)
    src = src[src[price_col] > 0.0]
    if src.empty:
        if default_value is None:
            raise ValueError("Cannot build historical sigma from empty close series.")
        starts = list(window_start_dates)
        return np.full((len(starts),), float(default_value), dtype=np.float64)

    prices = src[price_col].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(prices))
    if log_returns.size < int(window_days):
        if default_value is None:
            raise ValueError(
                f"Not enough log returns ({log_returns.size}) for rolling window={int(window_days)}."
            )
        starts = list(window_start_dates)
        return np.full((len(starts),), float(default_value), dtype=np.float64)

    sigma_ann = (
        pd.Series(log_returns)
        .rolling(window=int(window_days), min_periods=int(window_days))
        .std(ddof=1)
        .to_numpy(dtype=np.float64)
    ) * np.sqrt(float(trading_days_per_year))
    sigma_dates = src["Date"].iloc[1:].reset_index(drop=True)
    sigma_df = pd.DataFrame({"Date": sigma_dates, "sigma": sigma_ann})
    sigma_df = sigma_df.dropna(subset=["sigma"]).sort_values("Date").reset_index(drop=True)

    q_dates = list(window_start_dates)
    q = pd.DataFrame(
        {
            "query_idx": np.arange(len(q_dates), dtype=np.int64),
            "Date": pd.to_datetime(q_dates, errors="coerce"),
        }
    )

    if sigma_df.empty:
        if default_value is None:
            raise ValueError("Historical sigma series is empty after rolling estimation.")
        return np.full((q.shape[0],), float(default_value), dtype=np.float64)

    merged = pd.merge_asof(
        q.sort_values("Date"),
        sigma_df,
        on="Date",
        direction="backward",
        allow_exact_matches=False,
    )
    if merged["sigma"].isna().any():
        merged["sigma"] = merged["sigma"].ffill().bfill()
    if merged["sigma"].isna().any():
        if default_value is None:
            raise ValueError("Unable to map historical sigma to all window start dates.")
        merged["sigma"] = merged["sigma"].fillna(float(default_value))

    merged = merged.sort_values("query_idx")
    return merged["sigma"].to_numpy(dtype=np.float64)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _bs_price_scalar(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    S = float(S)
    K = float(K)
    T = max(float(T), 0.0)
    r = float(r)
    sigma = max(float(sigma), 1e-12)
    otype = str(option_type).strip().lower()
    if T <= 1e-12:
        if otype == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (math.log(max(S, 1e-12) / max(K, 1e-12)) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if otype == "call":
        return S * _normal_cdf(d1) - K * disc * _normal_cdf(d2)
    if otype == "put":
        return K * disc * _normal_cdf(-d2) - S * _normal_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'.")


def infer_implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    lower: float = 1e-6,
    upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 120,
) -> float:
    """
    Infer BS implied vol from option market price using bisection.
    Returns np.nan when no stable solution is found.
    """
    try:
        p_mkt = float(market_price)
        S = float(S)
        K = float(K)
        T = float(T)
        r = float(r)
    except Exception:
        return float("nan")

    if not np.isfinite(p_mkt) or p_mkt <= 0.0 or S <= 0.0 or K <= 0.0 or T <= 0.0:
        return float("nan")

    otype = str(option_type).strip().lower()
    disc = math.exp(-r * T)
    if otype == "call":
        intrinsic = max(S - K * disc, 0.0)
        max_price = S
    elif otype == "put":
        intrinsic = max(K * disc - S, 0.0)
        max_price = K * disc
    else:
        return float("nan")

    if p_mkt < intrinsic - 1e-8:
        return float("nan")
    if p_mkt > max_price + 1e-8:
        return float("nan")
    if abs(p_mkt - intrinsic) <= tol:
        return float(lower)

    low = float(lower)
    high = float(upper)
    p_low = _bs_price_scalar(S=S, K=K, T=T, r=r, sigma=low, option_type=otype)
    p_high = _bs_price_scalar(S=S, K=K, T=T, r=r, sigma=high, option_type=otype)

    if not (np.isfinite(p_low) and np.isfinite(p_high)):
        return float("nan")

    # Expand upper bound if needed.
    expand_iter = 0
    while p_high < p_mkt and expand_iter < 20:
        high *= 1.5
        p_high = _bs_price_scalar(S=S, K=K, T=T, r=r, sigma=high, option_type=otype)
        expand_iter += 1
    if p_high < p_mkt:
        return float("nan")

    for _ in range(int(max_iter)):
        mid = 0.5 * (low + high)
        p_mid = _bs_price_scalar(S=S, K=K, T=T, r=r, sigma=mid, option_type=otype)
        diff = p_mid - p_mkt
        if abs(diff) <= tol:
            return float(mid)
        if diff > 0.0:
            high = mid
        else:
            low = mid
    return float(0.5 * (low + high))


def _safe_option_market_price(row: pd.Series) -> float:
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    last = float(row.get("last", np.nan))
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0.0 and ask > 0.0 and ask >= bid:
        return float(0.5 * (bid + ask))
    if np.isfinite(last) and last > 0.0:
        return float(last)
    if np.isfinite(bid) and bid > 0.0:
        return float(bid)
    if np.isfinite(ask) and ask > 0.0:
        return float(ask)
    return float("nan")


def _parse_option_type_mask(series: pd.Series, option_type: str) -> pd.Series:
    vals = series.astype(str).str.strip().str.lower()
    otype = str(option_type).strip().lower()
    if otype == "call":
        accepted = {"call", "calls", "c"}
    elif otype == "put":
        accepted = {"put", "puts", "p"}
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    return vals.isin(accepted)


def _pick_quote_date_nearest_no_lookahead(
    dates_df: pd.DataFrame,
    start_date: pd.Timestamp,
    max_quote_lag_days: int,
) -> pd.Timestamp | None:
    if dates_df.empty:
        return None
    days_diff = (dates_df["quote_date"] - start_date).dt.days
    abs_days = days_diff.abs()
    candidates = dates_df.assign(_days_diff=days_diff, _abs_days=abs_days)
    candidates = candidates[candidates["_abs_days"] <= int(max_quote_lag_days)]
    if candidates.empty:
        return None
    # nearest quote; tie-breaker prefers same-day/previous quote (no look-ahead).
    candidates = candidates.sort_values(["_abs_days", "_days_diff"], ascending=[True, True])
    return pd.Timestamp(candidates.iloc[0]["quote_date"])


def map_option_market_implied_vol_to_window_start(
    option_quotes_csv: str,
    option_quote_date_col: str,
    option_expiry_col: str,
    option_strike_col: str,
    option_type_col: str,
    option_bid_col: str,
    option_ask_col: str,
    option_last_col: str,
    option_type: str,
    window_start_dates: Iterable[pd.Timestamp],
    window_start_spots: Iterable[float],
    per_path_r: Iterable[float],
    n_trading_days: int,
    trading_days_per_year: int,
    calendar_days_per_year: float,
    max_quote_lag_days: int,
    max_expiry_diff_days: int,
    default_sigma: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Map per-window implied sigma inferred from market option prices.
    Matching policy per window:
    1) quote date near window start (no-look-ahead tie break),
    2) expiration near target maturity,
    3) strike nearest spot at window start.
    """
    csv_path = os.path.abspath(os.path.expanduser(str(option_quotes_csv)))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"option_quotes_csv not found: {csv_path}")

    odf = pd.read_csv(csv_path)
    required_cols = [
        option_quote_date_col,
        option_expiry_col,
        option_strike_col,
        option_type_col,
        option_bid_col,
        option_ask_col,
        option_last_col,
    ]
    missing = [c for c in required_cols if c not in odf.columns]
    if missing:
        raise ValueError(f"option_quotes_csv missing required columns: {missing}")

    odf = odf.copy()
    odf["quote_date"] = pd.to_datetime(odf[option_quote_date_col], errors="coerce")
    odf["expiration"] = pd.to_datetime(odf[option_expiry_col], errors="coerce")
    odf["strike"] = pd.to_numeric(odf[option_strike_col], errors="coerce")
    odf["bid"] = pd.to_numeric(odf[option_bid_col], errors="coerce")
    odf["ask"] = pd.to_numeric(odf[option_ask_col], errors="coerce")
    odf["last"] = pd.to_numeric(odf[option_last_col], errors="coerce")
    odf = odf.dropna(subset=["quote_date", "expiration", "strike"]).reset_index(drop=True)
    odf = odf[_parse_option_type_mask(odf[option_type_col], option_type)]
    if odf.empty:
        raise ValueError("No rows left in option quotes after option_type filtering.")

    starts = pd.to_datetime(list(window_start_dates), errors="coerce")
    spots = np.asarray(list(window_start_spots), dtype=np.float64).reshape(-1)
    r_vec = np.asarray(list(per_path_r), dtype=np.float64).reshape(-1)
    if len(starts) != spots.shape[0] or len(starts) != r_vec.shape[0]:
        raise ValueError("window_start_dates, window_start_spots and per_path_r must have same length.")

    target_calendar_days = int(
        round(float(n_trading_days) * float(calendar_days_per_year) / float(trading_days_per_year))
    )
    default_sigma = float(default_sigma)
    if default_sigma <= 0.0 or not np.isfinite(default_sigma):
        raise ValueError(f"default_sigma must be finite and > 0. Got {default_sigma}")

    sigma_out = np.full((len(starts),), default_sigma, dtype=np.float64)
    rows = []
    for i, start_ts in enumerate(starts):
        spot_i = float(spots[i])
        r_i = float(r_vec[i])
        row_meta = {
            "window_idx": int(i),
            "window_start_date": pd.Timestamp(start_ts),
            "spot_start_raw": float(spot_i),
            "r_window": float(r_i),
            "target_calendar_days": int(target_calendar_days),
            "quote_date": pd.NaT,
            "expiration": pd.NaT,
            "days_to_exp": np.nan,
            "selected_strike": np.nan,
            "option_market_price": np.nan,
            "sigma_implied_from_option": float(default_sigma),
            "sigma_source_status": "fallback_default",
            "pricing_T_calendar": np.nan,
        }

        if not np.isfinite(spot_i) or spot_i <= 0.0 or pd.isna(start_ts):
            rows.append(row_meta)
            continue

        qdate = _pick_quote_date_nearest_no_lookahead(
            dates_df=odf[["quote_date"]].drop_duplicates(),
            start_date=pd.Timestamp(start_ts),
            max_quote_lag_days=int(max_quote_lag_days),
        )
        if qdate is None:
            rows.append(row_meta)
            continue

        day_slice = odf[odf["quote_date"] == qdate].copy()
        if day_slice.empty:
            rows.append(row_meta)
            continue

        day_slice["dte_days"] = (day_slice["expiration"] - day_slice["quote_date"]).dt.days
        day_slice = day_slice[day_slice["dte_days"] >= 1].copy()
        if day_slice.empty:
            rows.append(row_meta)
            continue

        day_slice["expiry_diff_days"] = (day_slice["dte_days"] - target_calendar_days).abs()
        day_slice = day_slice[day_slice["expiry_diff_days"] <= int(max_expiry_diff_days)].copy()
        if day_slice.empty:
            rows.append(row_meta)
            continue

        best_expiry = day_slice.sort_values(["expiry_diff_days", "dte_days"]).iloc[0]["expiration"]
        exp_slice = day_slice[day_slice["expiration"] == best_expiry].copy()
        exp_slice["strike_diff"] = (exp_slice["strike"] - spot_i).abs()
        best_row = exp_slice.sort_values(["strike_diff", "strike"]).iloc[0]

        market_price = _safe_option_market_price(best_row)
        K_sel = float(best_row["strike"])
        dte_days = int(best_row["dte_days"])
        t_cal = float(dte_days) / float(calendar_days_per_year)
        iv = infer_implied_volatility_bisection(
            market_price=market_price,
            S=spot_i,
            K=K_sel,
            T=t_cal,
            r=r_i,
            option_type=option_type,
        )
        if np.isfinite(iv) and iv > 0.0:
            sigma_out[i] = float(iv)
            row_meta["sigma_source_status"] = "option_implied"
        else:
            row_meta["sigma_source_status"] = "fallback_default"

        row_meta["quote_date"] = pd.Timestamp(qdate)
        row_meta["expiration"] = pd.Timestamp(best_expiry)
        row_meta["days_to_exp"] = int(dte_days)
        row_meta["selected_strike"] = float(K_sel)
        row_meta["option_market_price"] = float(market_price) if np.isfinite(market_price) else np.nan
        row_meta["pricing_T_calendar"] = float(t_cal)
        row_meta["sigma_implied_from_option"] = float(sigma_out[i])
        rows.append(row_meta)

    details_df = pd.DataFrame(rows)
    return sigma_out, details_df
