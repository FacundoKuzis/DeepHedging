import os

import numpy as np
import pandas as pd

from DeepHedging.utils.historical_windows import build_historical_windows_from_csv, to_environment_paths



def test_build_historical_windows_and_metadata(tmp_path):
    csv_path = os.path.join(str(tmp_path), "prices.csv")
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=7, freq="D"),
            "Close": [100, 101, 102, 103, 104, 105, 106],
        }
    )
    df.to_csv(csv_path, index=False)

    windows, meta = build_historical_windows_from_csv(
        csv_path=csv_path,
        n_hedging_steps=2,
        start_date="2020-01-01",
        end_date="2020-01-07",
        price_col="Close",
        stride=1,
        max_windows=None,
    )

    # 7 prices, window len 3 => 5 windows
    assert windows.shape == (5, 3)
    assert meta.shape[0] == 5
    assert np.isclose(windows[0, 0], 100.0)
    assert np.isclose(windows[-1, -1], 106.0)

    env_paths = to_environment_paths(windows)
    assert env_paths.shape == (5, 3, 1)
    assert env_paths.dtype == np.float32
