import os

import numpy as np
import pandas as pd

from DeepHedging.utils.gbm_calibration import calibrate_gbm_from_market_data, map_series_to_window_start



def _write_close_csv(path, prices):
    dates = pd.date_range("2000-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame({"Date": dates, "Close": prices})
    df.to_csv(path, index=False)



def test_calibrate_gbm_historical_sigma_fixed_risk_free(tmp_path):
    cache_dir = str(tmp_path)
    ticker = "TEST"
    start = "2000-01-01"
    end = "2000-01-10"
    interval = "1d"

    csv_path = os.path.join(cache_dir, f"{ticker}_{start}_{end}_{interval}.csv")
    _write_close_csv(csv_path, [100, 101, 99, 102, 103, 104, 102, 103, 105, 106])

    calib = calibrate_gbm_from_market_data(
        market_cache_dir=cache_dir,
        ticker=ticker,
        train_start_date=start,
        train_end_date=end,
        interval=interval,
        price_col="Close",
        trading_days_per_year=252,
        sigma_source="historical",
        implied_vol_source="fixed",
        implied_vol_stat="mean",
        fixed_implied_vol=0.2,
        risk_free_source="fixed",
        fixed_risk_free=0.01,
        download_if_missing=False,
    )

    assert calib.sigma_train > 0.0
    assert np.isclose(calib.r_train, 0.01)
    assert np.isclose(calib.mu_train, calib.r_train)



def test_calibrate_gbm_implied_fixed_vol(tmp_path):
    cache_dir = str(tmp_path)
    ticker = "TEST"
    start = "2000-01-01"
    end = "2000-01-10"
    interval = "1d"

    csv_path = os.path.join(cache_dir, f"{ticker}_{start}_{end}_{interval}.csv")
    _write_close_csv(csv_path, [100, 100.5, 100.2, 100.8, 101.0, 101.3, 101.1, 101.6, 101.9, 102.0])

    calib = calibrate_gbm_from_market_data(
        market_cache_dir=cache_dir,
        ticker=ticker,
        train_start_date=start,
        train_end_date=end,
        interval=interval,
        price_col="Close",
        trading_days_per_year=252,
        sigma_source="implied",
        implied_vol_source="fixed",
        implied_vol_stat="mean",
        fixed_implied_vol=0.25,
        risk_free_source="fixed",
        fixed_risk_free=0.02,
        download_if_missing=False,
    )

    assert np.isclose(calib.sigma_train, 0.25)
    assert np.isclose(calib.r_train, 0.02)



def test_map_series_to_window_start_ffill():
    series_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-08"]),
            "Close": [1.0, 2.0, 4.0],
        }
    )
    starts = pd.to_datetime(["2020-01-03", "2020-01-07", "2020-01-09"])
    vals = map_series_to_window_start(series_df, starts, value_col="Close", scale=1.0, default_value=0.0)
    assert vals.shape == (3,)
    np.testing.assert_allclose(vals, np.array([1.0, 2.0, 4.0]))
