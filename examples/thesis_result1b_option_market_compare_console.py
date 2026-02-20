"""
Console runner to compare theoretical European option prices vs historical
market option quotes window-by-window.

The script:
1) Builds historical windows from underlying prices.
2) Calibrates/loads per-window risk-free and volatility inputs.
3) Selects market option quotes per window:
   - quote date near window start
   - expiration near target maturity
   - strike nearest window-start spot
4) Computes Black-Scholes prices with:
   - T = N/252 (model convention)
   - T = calendar days to selected expiration / calendar_days_per_year
5) Saves detailed tables and plots.
"""

import argparse
import json
import math
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples.thesis_result1_common import (  # noqa: E402
    THESIS_MODELS_ROOT,
    copy_config_snapshot,
    load_config_by_name,
    strict_validate_keys,
)
from DeepHedging.utils.gbm_calibration import (  # noqa: E402
    IRX_TICKER,
    VIX_TICKER,
    calibrate_gbm_from_market_data,
    load_external_series,
    map_historical_sigma_to_window_start,
    map_series_to_window_start,
)
from DeepHedging.utils.historical_windows import (  # noqa: E402
    build_historical_windows_from_csv,
)


RESULT1B_ROOT = os.path.join(THESIS_MODELS_ROOT, "thesis_result1b")


def _run_dirs(run_name: str) -> dict[str, str]:
    run_dir = os.path.join(RESULT1B_ROOT, "option_market_compare", run_name)
    tables_dir = os.path.join(run_dir, "tables")
    plots_dir = os.path.join(run_dir, "plots")
    market_cache_dir = os.path.join(RESULT1B_ROOT, "market_data_cache")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(market_cache_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "market_cache_dir": market_cache_dir,
    }


def required_keys() -> set[str]:
    return {
        "schema_version",
        "model_family",
        "run_name",
        "description",
        "ticker",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "interval",
        "price_col",
        "download_if_missing",
        "n",
        "trading_days_per_year",
        "calendar_days_per_year",
        "historical_stride",
        "max_windows",
        "option_type",
        "option_quotes_csv",
        "option_quote_date_col",
        "option_expiry_col",
        "option_strike_col",
        "option_type_col",
        "option_bid_col",
        "option_ask_col",
        "option_last_col",
        "max_quote_lag_days",
        "max_expiry_diff_days",
        "sigma_source",
        "implied_vol_source",
        "implied_vol_stat",
        "fixed_implied_vol",
        "sigma_mode",
        "historical_sigma_window_days",
        "risk_free_source",
        "fixed_risk_free",
        "risk_free_mode",
    }


def optional_keys() -> set[str]:
    return set()


def _load_external_series_candidates(
    market_cache_dir: str,
    candidates: list[str],
    start_date: str,
    end_date: str,
    interval: str,
    download_if_missing: bool,
    price_col: str = "Close",
):
    last_exc = None
    for ticker_candidate in candidates:
        try:
            return load_external_series(
                market_cache_dir=market_cache_dir,
                ticker=ticker_candidate,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                download_if_missing=download_if_missing,
                price_col=price_col,
            ), ticker_candidate
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise ValueError(
            f"Unable to load external series from candidates={candidates}. Last error: {last_exc}"
        ) from last_exc
    raise ValueError(f"Unable to load external series from candidates={candidates}.")


def _norm_text(x: Any) -> str:
    return str(x).strip().lower()


def validate_config(config: dict[str, Any]) -> None:
    strict_validate_keys(config, required_keys(), optional_keys())
    if int(config["schema_version"]) != 1:
        raise ValueError("schema_version must be 1.")
    if _norm_text(config["model_family"]) != "deep_hedging_result_1b_option_market_compare":
        raise ValueError("model_family must be 'deep_hedging_result_1b_option_market_compare'.")
    for key in [
        "run_name",
        "description",
        "ticker",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "interval",
        "price_col",
        "option_quotes_csv",
        "option_quote_date_col",
        "option_expiry_col",
        "option_strike_col",
        "option_type_col",
        "option_bid_col",
        "option_ask_col",
        "option_last_col",
    ]:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be non-empty string.")
    if not os.path.isfile(str(config["option_quotes_csv"])):
        raise FileNotFoundError(f"option_quotes_csv not found: {config['option_quotes_csv']}")

    if int(config["n"]) <= 0:
        raise ValueError("n must be > 0.")
    if int(config["trading_days_per_year"]) <= 0:
        raise ValueError("trading_days_per_year must be > 0.")
    if float(config["calendar_days_per_year"]) <= 0.0:
        raise ValueError("calendar_days_per_year must be > 0.")
    if int(config["historical_stride"]) <= 0:
        raise ValueError("historical_stride must be > 0.")
    if config["max_windows"] is not None and int(config["max_windows"]) <= 0:
        raise ValueError("max_windows must be null or > 0.")
    if int(config["max_quote_lag_days"]) < 0:
        raise ValueError("max_quote_lag_days must be >= 0.")
    if int(config["max_expiry_diff_days"]) < 0:
        raise ValueError("max_expiry_diff_days must be >= 0.")
    if int(config["historical_sigma_window_days"]) < 2:
        raise ValueError("historical_sigma_window_days must be >= 2.")
    if _norm_text(config["option_type"]) not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if not isinstance(config["download_if_missing"], bool):
        raise ValueError("download_if_missing must be bool.")

    sigma_source = _norm_text(config["sigma_source"])
    sigma_mode = _norm_text(config["sigma_mode"])
    if sigma_source not in {"historical", "implied"}:
        raise ValueError("sigma_source must be 'historical' or 'implied'.")
    if sigma_mode not in {"train_average", "per_window_start", "rolling_pre_window"}:
        raise ValueError(
            "sigma_mode must be one of {'train_average','per_window_start','rolling_pre_window'}."
        )
    if sigma_mode == "rolling_pre_window" and sigma_source != "historical":
        raise ValueError("sigma_mode='rolling_pre_window' requires sigma_source='historical'.")
    if sigma_mode == "per_window_start" and sigma_source == "historical":
        raise ValueError(
            "sigma_mode='per_window_start' unsupported for sigma_source='historical'. "
            "Use 'rolling_pre_window' or 'train_average'."
        )
    implied_source = _norm_text(config["implied_vol_source"])
    if implied_source not in {"vix", "fixed"}:
        raise ValueError("implied_vol_source must be 'vix' or 'fixed'.")
    if _norm_text(config["implied_vol_stat"]) not in {"mean", "median"}:
        raise ValueError("implied_vol_stat must be 'mean' or 'median'.")
    if implied_source == "fixed":
        if config["fixed_implied_vol"] is None or float(config["fixed_implied_vol"]) <= 0.0:
            raise ValueError("fixed_implied_vol must be > 0 when implied_vol_source='fixed'.")

    risk_free_source = _norm_text(config["risk_free_source"])
    risk_free_mode = _norm_text(config["risk_free_mode"])
    if risk_free_source not in {"irx", "fixed"}:
        raise ValueError("risk_free_source must be 'irx' or 'fixed'.")
    if risk_free_mode not in {"train_average", "per_window_start"}:
        raise ValueError("risk_free_mode must be 'train_average' or 'per_window_start'.")
    if risk_free_source == "fixed":
        if config["fixed_risk_free"] is None:
            raise ValueError("fixed_risk_free must be provided when risk_free_source='fixed'.")


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    S = float(S)
    K = float(K)
    T = float(max(T, 0.0))
    r = float(r)
    sigma = float(max(sigma, 1e-10))
    option_type = _norm_text(option_type)
    if T <= 1e-12:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (np.log(max(S, 1e-12) / max(K, 1e-12)) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    n1 = float(_normal_cdf(np.array([d1]))[0])
    n2 = float(_normal_cdf(np.array([d2]))[0])
    disc = np.exp(-r * T)
    if option_type == "call":
        return S * n1 - K * disc * n2
    return K * disc * float(_normal_cdf(np.array([-d2]))[0]) - S * float(_normal_cdf(np.array([-d1]))[0])


def _parse_option_type_mask(series: pd.Series, option_type: str) -> pd.Series:
    target = _norm_text(option_type)
    vals = series.astype(str).str.strip().str.lower()
    if target == "call":
        accepted = {"call", "c", "calls"}
    else:
        accepted = {"put", "p", "puts"}
    return vals.isin(accepted)


def _pick_quote_date(df: pd.DataFrame, start_date: pd.Timestamp, max_lag_days: int) -> pd.Timestamp | None:
    if df.empty:
        return None
    days_diff = (df["quote_date"] - start_date).dt.days
    abs_days = days_diff.abs()
    candidates = df.assign(_days_diff=days_diff, _abs_days=abs_days)
    candidates = candidates[candidates["_abs_days"] <= int(max_lag_days)]
    if candidates.empty:
        return None
    # Prefer same-day or previous quote to avoid look-ahead bias.
    # If both are equally close, a negative day-diff (previous day) is selected.
    candidates = candidates.sort_values(["_abs_days", "_days_diff"], ascending=[True, True])
    return pd.Timestamp(candidates.iloc[0]["quote_date"])


def _safe_market_price(row: pd.Series) -> float:
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    last = float(row.get("last", np.nan))
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0.0 and ask > 0.0 and ask >= bid:
        return 0.5 * (bid + ask)
    if np.isfinite(last) and last > 0.0:
        return last
    if np.isfinite(bid) and bid > 0.0:
        return bid
    if np.isfinite(ask) and ask > 0.0:
        return ask
    return np.nan


def _summary_metrics(df: pd.DataFrame, market_col: str, model_col: str) -> dict[str, float]:
    tmp = df[[market_col, model_col]].dropna()
    if tmp.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape_pct": np.nan,
            "corr": np.nan,
            "mean_model_minus_market": np.nan,
        }
    market = tmp[market_col].to_numpy(dtype=np.float64)
    model = tmp[model_col].to_numpy(dtype=np.float64)
    err = model - market
    mape = np.mean(np.abs(err) / np.maximum(np.abs(market), 1e-8)) * 100.0
    corr = np.corrcoef(market, model)[0, 1] if market.size > 1 else np.nan
    return {
        "n": int(market.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(mape),
        "corr": float(corr),
        "mean_model_minus_market": float(np.mean(err)),
    }


def run_comparison(run_name: str, config_path: str, config: dict[str, Any]) -> None:
    dirs = _run_dirs(run_name)
    copied = copy_config_snapshot(config_path, dirs["run_dir"])
    print(f"[run:{run_name}] Config validated and copied to: {copied}")

    calib = calibrate_gbm_from_market_data(
        market_cache_dir=dirs["market_cache_dir"],
        ticker=str(config["ticker"]),
        train_start_date=str(config["train_start_date"]),
        train_end_date=str(config["train_end_date"]),
        interval=str(config["interval"]),
        price_col=str(config["price_col"]),
        trading_days_per_year=int(config["trading_days_per_year"]),
        sigma_source=str(config["sigma_source"]),
        implied_vol_source=str(config["implied_vol_source"]),
        implied_vol_stat=str(config["implied_vol_stat"]),
        fixed_implied_vol=None if config["fixed_implied_vol"] is None else float(config["fixed_implied_vol"]),
        risk_free_source=str(config["risk_free_source"]),
        fixed_risk_free=None if config["fixed_risk_free"] is None else float(config["fixed_risk_free"]),
        download_if_missing=bool(config["download_if_missing"]),
    )

    test_csv = os.path.join(
        dirs["market_cache_dir"],
        f"{config['ticker']}_{config['test_start_date']}_{config['test_end_date']}_{config['interval']}.csv",
    )
    if not os.path.isfile(test_csv):
        from DeepHedging.utils.market_data import download_ohlcv_to_csv

        download_ohlcv_to_csv(
            ticker=str(config["ticker"]),
            start_date=str(config["test_start_date"]),
            end_date=str(config["test_end_date"]),
            interval=str(config["interval"]),
            output_csv=test_csv,
        )

    windows_2d, windows_meta = build_historical_windows_from_csv(
        csv_path=test_csv,
        n_hedging_steps=int(config["n"]),
        start_date=str(config["test_start_date"]),
        end_date=str(config["test_end_date"]),
        price_col=str(config["price_col"]),
        stride=int(config["historical_stride"]),
        max_windows=None if config["max_windows"] is None else int(config["max_windows"]),
    )
    starts = windows_meta["start_date"].tolist()
    spot_raw = windows_2d[:, 0].astype(np.float64)

    risk_free_mode = _norm_text(config["risk_free_mode"])
    if _norm_text(config["risk_free_source"]) == "fixed":
        per_path_r = np.full((len(starts),), float(config["fixed_risk_free"]), dtype=np.float64)
    elif risk_free_mode == "train_average":
        per_path_r = np.full((len(starts),), float(calib.r_train), dtype=np.float64)
    else:
        try:
            irx_df, irx_ticker = _load_external_series_candidates(
                market_cache_dir=dirs["market_cache_dir"],
                candidates=[IRX_TICKER, "IRX", "^FVX", "^TNX"],
                start_date=str(config["test_start_date"]),
                end_date=str(config["test_end_date"]),
                interval=str(config["interval"]),
                download_if_missing=bool(config["download_if_missing"]),
                price_col="Close",
            )
            irx_scale = 0.01 if irx_ticker in {IRX_TICKER, "IRX"} else 0.001
            per_path_r = map_series_to_window_start(
                series_df=irx_df,
                window_start_dates=starts,
                value_col="Close",
                scale=float(irx_scale),
                default_value=float(calib.r_train),
            )
        except Exception as exc:
            print(f"[run:{run_name}] [warning] per-window risk-free failed, fallback to train-average. Detail: {exc}")
            per_path_r = np.full((len(starts),), float(calib.r_train), dtype=np.float64)

    sigma_source = _norm_text(config["sigma_source"])
    sigma_mode = _norm_text(config["sigma_mode"])
    if sigma_mode == "train_average":
        per_path_sigma = np.full((len(starts),), float(calib.sigma_train), dtype=np.float64)
    elif sigma_source == "implied" and sigma_mode == "per_window_start":
        implied_source = _norm_text(config["implied_vol_source"])
        if implied_source == "vix":
            try:
                vix_df, _ = _load_external_series_candidates(
                    market_cache_dir=dirs["market_cache_dir"],
                    candidates=[VIX_TICKER, "VIX"],
                    start_date=str(config["test_start_date"]),
                    end_date=str(config["test_end_date"]),
                    interval=str(config["interval"]),
                    download_if_missing=bool(config["download_if_missing"]),
                    price_col="Close",
                )
                per_path_sigma = map_series_to_window_start(
                    series_df=vix_df,
                    window_start_dates=starts,
                    value_col="Close",
                    scale=0.01,
                    default_value=float(calib.sigma_train),
                )
            except Exception as exc:
                print(f"[run:{run_name}] [warning] per-window implied sigma failed, fallback to train-average. Detail: {exc}")
                per_path_sigma = np.full((len(starts),), float(calib.sigma_train), dtype=np.float64)
        else:
            per_path_sigma = np.full((len(starts),), float(calib.sigma_train), dtype=np.float64)
    elif sigma_source == "historical" and sigma_mode == "rolling_pre_window":
        full_close_df = load_external_series(
            market_cache_dir=dirs["market_cache_dir"],
            ticker=str(config["ticker"]),
            start_date=str(config["train_start_date"]),
            end_date=str(config["test_end_date"]),
            interval=str(config["interval"]),
            download_if_missing=bool(config["download_if_missing"]),
            price_col=str(config["price_col"]),
        )
        per_path_sigma = map_historical_sigma_to_window_start(
            close_df=full_close_df,
            window_start_dates=starts,
            price_col=str(config["price_col"]),
            window_days=int(config["historical_sigma_window_days"]),
            trading_days_per_year=int(config["trading_days_per_year"]),
            default_value=float(calib.sigma_train),
        )
    else:
        per_path_sigma = np.full((len(starts),), float(calib.sigma_train), dtype=np.float64)

    options_csv = os.path.abspath(os.path.expanduser(str(config["option_quotes_csv"])))
    print(f"[run:{run_name}] Option quotes CSV: {options_csv}")
    odf = pd.read_csv(options_csv)
    date_col = str(config["option_quote_date_col"])
    exp_col = str(config["option_expiry_col"])
    strike_col = str(config["option_strike_col"])
    type_col = str(config["option_type_col"])
    bid_col = str(config["option_bid_col"])
    ask_col = str(config["option_ask_col"])
    last_col = str(config["option_last_col"])
    for col in [date_col, exp_col, strike_col, type_col, bid_col, ask_col, last_col]:
        if col not in odf.columns:
            raise ValueError(f"option_quotes_csv is missing required column '{col}'.")

    odf = odf.copy()
    odf["quote_date"] = pd.to_datetime(odf[date_col], errors="coerce")
    odf["expiration"] = pd.to_datetime(odf[exp_col], errors="coerce")
    odf["strike"] = pd.to_numeric(odf[strike_col], errors="coerce")
    odf["bid"] = pd.to_numeric(odf[bid_col], errors="coerce")
    odf["ask"] = pd.to_numeric(odf[ask_col], errors="coerce")
    odf["last"] = pd.to_numeric(odf[last_col], errors="coerce")
    odf = odf.dropna(subset=["quote_date", "expiration", "strike"]).reset_index(drop=True)
    odf = odf[_parse_option_type_mask(odf[type_col], _norm_text(config["option_type"]))]
    if odf.empty:
        raise ValueError("No option rows left after filtering by option_type.")

    t_252 = float(config["n"]) / float(config["trading_days_per_year"])
    target_calendar_days = int(round(float(config["n"]) * float(config["calendar_days_per_year"]) / float(config["trading_days_per_year"])))
    max_quote_lag = int(config["max_quote_lag_days"])
    max_exp_diff = int(config["max_expiry_diff_days"])

    print(
        f"[run:{run_name}] Matching policy: strike closest to spot_start, quote_date within +/-{max_quote_lag} days, "
        f"expiry within +/-{max_exp_diff} days of {target_calendar_days} target calendar days."
    )
    print(
        f"[run:{run_name}] Pricing conventions: bs_252 uses T={t_252:.6f}, "
        "bs_calendar uses T=days_to_exp/calendar_days_per_year."
    )

    rows = []
    total = len(starts)
    for i, start_date in enumerate(starts):
        start_ts = pd.Timestamp(start_date)
        spot_i = float(spot_raw[i])
        r_i = float(per_path_r[i])
        sigma_i = float(per_path_sigma[i])

        qdate = _pick_quote_date(odf, start_ts, max_quote_lag)
        if qdate is None:
            continue

        day_slice = odf[odf["quote_date"] == qdate].copy()
        if day_slice.empty:
            continue

        day_slice["dte_days"] = (day_slice["expiration"] - qdate).dt.days
        day_slice = day_slice[day_slice["dte_days"] >= 1].copy()
        if day_slice.empty:
            continue

        day_slice["expiry_diff_days"] = (day_slice["dte_days"] - target_calendar_days).abs()
        day_slice = day_slice[day_slice["expiry_diff_days"] <= max_exp_diff].copy()
        if day_slice.empty:
            continue

        # First pick expiration nearest target maturity, then strike nearest spot.
        best_expiry = day_slice.sort_values(["expiry_diff_days", "dte_days"]).iloc[0]["expiration"]
        exp_slice = day_slice[day_slice["expiration"] == best_expiry].copy()
        exp_slice["strike_diff"] = (exp_slice["strike"] - spot_i).abs()
        best_row = exp_slice.sort_values(["strike_diff", "strike"]).iloc[0]

        market_price = _safe_market_price(best_row)
        if not np.isfinite(market_price) or market_price <= 0.0:
            continue

        K_sel = float(best_row["strike"])
        dte_days = int(best_row["dte_days"])
        t_cal = max(float(dte_days) / float(config["calendar_days_per_year"]), 1e-8)
        bs_252 = _bs_price(spot_i, K_sel, t_252, r_i, sigma_i, _norm_text(config["option_type"]))
        bs_cal = _bs_price(spot_i, K_sel, t_cal, r_i, sigma_i, _norm_text(config["option_type"]))

        rows.append(
            {
                "window_idx": int(i),
                "window_start_date": start_ts,
                "window_end_date": pd.Timestamp(windows_meta.iloc[i]["end_date"]),
                "spot_start_raw": float(spot_i),
                "quote_date": pd.Timestamp(qdate),
                "expiration": pd.Timestamp(best_expiry),
                "days_to_exp": int(dte_days),
                "target_calendar_days": int(target_calendar_days),
                "selected_strike": float(K_sel),
                "strike_minus_spot": float(K_sel - spot_i),
                "risk_free_used": float(r_i),
                "sigma_used": float(sigma_i),
                "market_option_price": float(market_price),
                "bs_price_252": float(bs_252),
                "bs_price_calendar": float(bs_cal),
                "market_option_price_norm": float(market_price / max(spot_i, 1e-12)),
                "bs_price_252_norm": float(bs_252 / max(spot_i, 1e-12)),
                "bs_price_calendar_norm": float(bs_cal / max(spot_i, 1e-12)),
                "error_252": float(bs_252 - market_price),
                "error_calendar": float(bs_cal - market_price),
            }
        )

        if (i + 1) % 250 == 0 or (i + 1) == total:
            print(f"[run:{run_name}] Processed {i+1}/{total} windows. Matched so far: {len(rows)}")

    if not rows:
        raise ValueError(
            "No window could be matched to market options with current constraints. "
            "Relax max_quote_lag_days/max_expiry_diff_days or check option CSV coverage."
        )

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(dirs["tables_dir"], "window_level_option_market_comparison.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[run:{run_name}] Saved window-level comparison: {out_csv}")

    summary_rows = []
    for market_col, model_col, label in [
        ("market_option_price", "bs_price_252", "bs_252_vs_market"),
        ("market_option_price", "bs_price_calendar", "bs_calendar_vs_market"),
        ("market_option_price_norm", "bs_price_252_norm", "bs_252_norm_vs_market_norm"),
        ("market_option_price_norm", "bs_price_calendar_norm", "bs_calendar_norm_vs_market_norm"),
    ]:
        m = _summary_metrics(out_df, market_col, model_col)
        m["comparison"] = label
        summary_rows.append(m)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(dirs["tables_dir"], "summary_option_market_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[run:{run_name}] Saved summary metrics: {summary_csv}")

    plt.figure(figsize=(8, 6))
    plt.scatter(out_df["market_option_price"], out_df["bs_price_252"], s=10, alpha=0.5)
    lim_low = float(min(out_df["market_option_price"].min(), out_df["bs_price_252"].min()))
    lim_high = float(max(out_df["market_option_price"].max(), out_df["bs_price_252"].max()))
    plt.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=1)
    plt.xlabel("Precio mercado")
    plt.ylabel("Precio BS (T=N/252)")
    plt.title("Mercado vs BS (convencion 252)")
    plt.grid(True, alpha=0.3)
    p1 = os.path.join(dirs["plots_dir"], "market_vs_bs_252.jpg")
    plt.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(out_df["market_option_price"], out_df["bs_price_calendar"], s=10, alpha=0.5)
    lim_low = float(min(out_df["market_option_price"].min(), out_df["bs_price_calendar"].min()))
    lim_high = float(max(out_df["market_option_price"].max(), out_df["bs_price_calendar"].max()))
    plt.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=1)
    plt.xlabel("Precio mercado")
    plt.ylabel("Precio BS (T calendario)")
    plt.title("Mercado vs BS (vencimiento calendario)")
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(dirs["plots_dir"], "market_vs_bs_calendar.jpg")
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    x = pd.to_datetime(out_df["window_start_date"])
    plt.plot(x, out_df["error_252"], label="Error BS 252 - mercado", alpha=0.8)
    plt.plot(x, out_df["error_calendar"], label="Error BS calendario - mercado", alpha=0.8)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Fecha inicio ventana")
    plt.ylabel("Error de precio")
    plt.title("Error de pricing por ventana")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p3 = os.path.join(dirs["plots_dir"], "pricing_error_timeseries.jpg")
    plt.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"[run:{run_name}] Saved plots: {p1}, {p2}, {p3}")

    meta = {
        "run_name": run_name,
        "matched_windows": int(out_df.shape[0]),
        "total_windows": int(total),
        "target_calendar_days": int(target_calendar_days),
        "n_trading_days": int(config["n"]),
        "trading_days_per_year": int(config["trading_days_per_year"]),
        "calendar_days_per_year": float(config["calendar_days_per_year"]),
        "option_quotes_csv": str(config["option_quotes_csv"]),
        "sigma_source": str(config["sigma_source"]),
        "sigma_mode": str(config["sigma_mode"]),
        "risk_free_source": str(config["risk_free_source"]),
        "risk_free_mode": str(config["risk_free_mode"]),
    }
    meta_path = os.path.join(dirs["run_dir"], "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[run:{run_name}] Saved metadata: {meta_path}")
    print(f"[run:{run_name}] Completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare European option model prices vs market option quotes by historical window."
    )
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Optional config filename in ./thesis_result1b_configs/option_market_compare (extension optional).",
    )
    args = parser.parse_args()

    configs_dir = os.path.join(os.getcwd(), "thesis_result1b_configs", "option_market_compare")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"Config folder not found: {configs_dir}")

    run_name, cfg_path, cfg = load_config_by_name(
        configs_dir=configs_dir,
        config_name=args.config_name,
        prompt_label="Enter OPTION-MARKET-COMPARE JSON config name from 'thesis_result1b_configs/option_market_compare': ",
    )
    print(f"[run:{run_name}] Loaded config: {cfg_path}")
    validate_config(cfg)
    run_comparison(run_name=run_name, config_path=cfg_path, config=cfg)


if __name__ == "__main__":
    main()
