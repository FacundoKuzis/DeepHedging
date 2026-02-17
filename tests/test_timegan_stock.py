import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.HedgingInstruments import TimeGANStock


class TestTimeGANStock(unittest.TestCase):
    def _make_prices_csv(self, folder: str, rows: int = 300) -> str:
        csv_path = os.path.join(folder, "prices.csv")
        dates = pd.date_range("2020-01-01", periods=rows, freq="D")
        close = np.linspace(90.0, 130.0, rows)
        df = pd.DataFrame({"Date": dates, "Close": close})
        df.to_csv(csv_path, index=False)
        return csv_path

    def _build_instrument(self, csv_path: str) -> TimeGANStock:
        return TimeGANStock(
            S0=100.0,
            T=63 / 252,
            N=63,
            r=0.05,
            ticker="SPY",
            start_date="2019-01-01",
            end_date="2024-12-31",
            interval="1d",
            price_col="Close",
            csv_path=csv_path,
            model_path=os.path.join(os.path.dirname(csv_path), "model.pkl"),
            download_if_missing=False,
            retrain=False,
            stride=1,
            random_seed=7,
            min_windows=50,
            train_epochs=1,
        )

    def test_windows_to_long_format_cardinality(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_prices_csv(tmp)
            instrument = self._build_instrument(csv_path)

            subset = instrument._real_windows[:3]
            long_df = instrument._windows_to_long_format(subset)

            self.assertListEqual(list(long_df.columns), ["entity_id", "time_idx", "close"])
            self.assertEqual(len(long_df), 3 * (instrument.N + 1))
            self.assertEqual(long_df["entity_id"].nunique(), 3)
            self.assertEqual(long_df["time_idx"].nunique(), instrument.N + 1)

    def test_generate_paths_shape_dtype_and_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_prices_csv(tmp)
            instrument = self._build_instrument(csv_path)

            mocked_window = np.linspace(0.4, 0.6, instrument.N, dtype=np.float32)
            mocked_samples = np.tile(mocked_window, (5, 1))
            with patch.object(instrument, "_sample_windows", return_value=mocked_samples):
                paths = instrument.generate_paths(num_paths=5, random_seed=123)

            self.assertEqual(tuple(paths.shape), (5, instrument.N + 1))
            self.assertEqual(paths.dtype.name, "float32")
            np.testing.assert_allclose(paths.numpy()[:, 0], np.full((5,), instrument.S0), rtol=0, atol=1e-6)

    def test_postprocess_legacy_price_levels_still_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_prices_csv(tmp)
            instrument = self._build_instrument(csv_path)

            legacy_scaled = np.tile(
                np.linspace(0.2, 0.8, instrument.N + 1, dtype=np.float32),
                (2, 1),
            )
            prices = instrument._postprocess_to_prices(legacy_scaled)
            self.assertEqual(prices.shape, (2, instrument.N + 1))
            self.assertTrue(np.all(prices > 0))
            np.testing.assert_allclose(prices[:, 0], np.full((2,), instrument.S0), rtol=0, atol=1e-6)

    def test_postprocess_enforces_positivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_prices_csv(tmp)
            instrument = self._build_instrument(csv_path)

            bad = np.array([[-1.0] + [0.0] * instrument.N], dtype=np.float32)
            fixed = instrument._postprocess_to_prices(bad)
            self.assertTrue(np.all(fixed > 0))
            self.assertAlmostEqual(float(fixed[0, 0]), instrument.S0, places=6)

    def test_raises_when_csv_missing_and_download_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_csv = os.path.join(tmp, "does_not_exist.csv")
            with self.assertRaises(FileNotFoundError):
                TimeGANStock(
                    S0=100.0,
                    T=63 / 252,
                    N=63,
                    r=0.05,
                    ticker="SPY",
                    start_date="2019-01-01",
                    end_date="2024-12-31",
                    csv_path=missing_csv,
                    model_path=os.path.join(tmp, "model.pkl"),
                    download_if_missing=False,
                    min_windows=10,
                )

    def test_raises_when_ydata_dependency_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_prices_csv(tmp)
            instrument = self._build_instrument(csv_path)

            with patch.object(
                TimeGANStock,
                "_require_ydata",
                side_effect=ImportError("missing ydata-synthetic"),
            ):
                with self.assertRaises(ImportError):
                    instrument._fit_or_load_synthesizer()


if __name__ == "__main__":
    unittest.main()
