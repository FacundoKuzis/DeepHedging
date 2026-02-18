import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.HedgingInstruments import TimeGANStock


class TestTimeganWindowingAndManifest(unittest.TestCase):
    def _make_prices_csv(self, folder: str, rows: int = 600) -> str:
        csv_path = os.path.join(folder, "prices.csv")
        dates = pd.date_range("2010-01-01", periods=rows, freq="D")
        close = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, size=rows))
        close = np.maximum(close, 1.0)
        pd.DataFrame({"Date": dates, "Close": close}).to_csv(csv_path, index=False)
        return csv_path

    def test_explicit_windows_tensor_shape_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_prices_csv(tmp, rows=500)
            instrument = TimeGANStock(
                S0=100.0,
                T=22 / 252,
                N=22,
                r=0.05,
                ticker="TEST",
                start_date="2010-01-01",
                end_date="2011-12-31",
                csv_path=csv_path,
                model_path=os.path.join(tmp, "model.pkl"),
                download_if_missing=False,
                retrain=False,
                stride=2,
                min_windows=50,
                train_epochs=1,
                fit_input_mode="explicit_windows",
                return_transform="gaussian_cdf",
                feature_mode="returns_plus_abs_return",
                feature_rolling_vol_window=5,
            )
            tensor = instrument._training_tensor
            self.assertIsNotNone(tensor)
            self.assertEqual(tensor.ndim, 3)
            self.assertEqual(tensor.shape[1], 22)
            self.assertEqual(tensor.shape[2], 2)
            self.assertGreaterEqual(tensor.shape[0], 50)

            manifest = instrument.get_training_manifest()
            self.assertFalse(manifest.empty)
            self.assertIn("n_windows", manifest.columns)
            self.assertIn("feature_transform", manifest.columns)
            self.assertTrue((manifest["fit_input_mode"] == "explicit_windows").all())


if __name__ == "__main__":
    unittest.main()
