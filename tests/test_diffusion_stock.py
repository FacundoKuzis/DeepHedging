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

from DeepHedging.HedgingInstruments import DiffusionStock


class TestDiffusionStock(unittest.TestCase):
    def _make_prices_csv(self, folder: str, rows: int = 800) -> str:
        csv_path = os.path.join(folder, "prices.csv")
        dates = pd.date_range("2010-01-01", periods=rows, freq="D")
        rng = np.random.default_rng(123)
        log_r = rng.normal(loc=0.0001, scale=0.01, size=rows - 1)
        prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(log_r))])
        prices = np.maximum(prices, 1.0)
        pd.DataFrame({"Date": dates, "Close": prices}).to_csv(csv_path, index=False)
        return csv_path

    def test_generate_paths_and_reuse_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_prices_csv(tmp)
            model_dir = os.path.join(tmp, "model")

            instrument = DiffusionStock(
                S0=100.0,
                T=22 / 252,
                N=22,
                r=0.05,
                ticker="TEST",
                start_date="2010-01-01",
                end_date="2012-12-31",
                csv_path=csv_path,
                model_dir=model_dir,
                download_if_missing=False,
                retrain=True,
                stride=2,
                min_windows=50,
                train_epochs=2,
                batch_size=32,
                learning_rate=2e-4,
                weight_decay=0.0,
                grad_clip_norm=1.0,
                use_ema=True,
                ema_decay=0.99,
                return_transform="gaussian_cdf",
                feature_mode="returns_only",
                diffusion_steps=20,
                beta_schedule="linear",
                beta_start=1e-4,
                beta_end=0.02,
                model_hidden_dim=32,
                model_num_res_blocks=2,
                model_dropout=0.1,
                time_embedding_dim=16,
                sampler_type="ddpm",
                sample_steps=20,
            )

            paths = instrument.generate_paths(num_paths=8, random_seed=7).numpy()
            self.assertEqual(paths.shape, (8, 23))
            self.assertEqual(paths.dtype.name, "float32")
            np.testing.assert_allclose(paths[:, 0], np.full((8,), 100.0), atol=1e-6)
            self.assertTrue(np.isfinite(paths).all())

            self.assertTrue(os.path.exists(os.path.join(model_dir, "diffusion_denoiser.keras")))
            self.assertTrue(os.path.exists(os.path.join(model_dir, "diffusion_state.npz")))
            self.assertTrue(os.path.exists(os.path.join(model_dir, "diffusion_metadata.json")))

            reused = DiffusionStock(
                S0=100.0,
                T=22 / 252,
                N=22,
                r=0.05,
                ticker="TEST",
                start_date="2010-01-01",
                end_date="2012-12-31",
                csv_path=csv_path,
                model_dir=model_dir,
                download_if_missing=False,
                retrain=False,
                stride=2,
                min_windows=50,
                train_epochs=2,
                batch_size=32,
                learning_rate=2e-4,
                weight_decay=0.0,
                grad_clip_norm=1.0,
                use_ema=True,
                ema_decay=0.99,
                return_transform="gaussian_cdf",
                feature_mode="returns_only",
                diffusion_steps=20,
                beta_schedule="linear",
                beta_start=1e-4,
                beta_end=0.02,
                model_hidden_dim=32,
                model_num_res_blocks=2,
                model_dropout=0.1,
                time_embedding_dim=16,
                sampler_type="ddpm",
                sample_steps=20,
            )
            paths_reuse = reused.generate_paths(num_paths=4, random_seed=9).numpy()
            self.assertEqual(paths_reuse.shape, (4, 23))


if __name__ == "__main__":
    unittest.main()
