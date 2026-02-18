import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from examples.diffusion_simulator_console import run_simulation, validate_config


class TestDiffusionRunnerSmoke(unittest.TestCase):
    def _write_prices_csv(self, path: str, rows: int = 180) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dates = pd.date_range("2010-01-01", periods=rows, freq="D")
        rng = np.random.default_rng(77)
        log_r = rng.normal(loc=0.0002, scale=0.01, size=rows - 1)
        prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(log_r))])
        pd.DataFrame({"Date": dates, "Close": prices}).to_csv(path, index=False)

    def test_run_simulation_smoke(self):
        prev_output_root = os.environ.get("TIMEGAN_OUTPUT_ROOT")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output_root = os.path.join(tmp, "outputs")
                os.makedirs(output_root, exist_ok=True)
                os.environ["TIMEGAN_OUTPUT_ROOT"] = output_root

                market_cache = os.path.join(output_root, "market_data_cache")
                train_csv = os.path.join(market_cache, "TEST_2010-01-01_2010-06-30_1d.csv")
                test_csv = os.path.join(market_cache, "TEST_2010-07-01_2010-12-31_1d.csv")
                self._write_prices_csv(train_csv, rows=180)
                self._write_prices_csv(test_csv, rows=180)

                config = {
                "schema_version": 1,
                "model_family": "diffusion",
                "s0": 100.0,
                "t": 8 / 252,
                "n": 8,
                "r": 0.01,
                "ticker": "TEST",
                "train_start_date": "2010-01-01",
                "train_end_date": "2010-06-30",
                "test_start_date": "2010-07-01",
                "test_end_date": "2010-12-31",
                "interval": "1d",
                "price_col": "Close",
                "download_if_missing": False,
                "retrain": True,
                "stride": 1,
                "min_windows": 5,
                "random_seed": 123,
                "training_target": "log_returns",
                "feature_mode": "returns_only",
                "feature_rolling_vol_window": 5,
                "return_transform": "gaussian_cdf",
                "transform_eps": 1e-6,
                "legacy_return_clip": False,
                "train_epochs": 1,
                "batch_size": 8,
                "learning_rate": 2e-4,
                "weight_decay": 0.0,
                "grad_clip_norm": 1.0,
                "use_ema": True,
                "ema_decay": 0.99,
                "diffusion_steps": 10,
                "beta_schedule": "linear",
                "beta_start": 1e-4,
                "beta_end": 0.02,
                "model_hidden_dim": 16,
                "model_num_res_blocks": 2,
                "model_dropout": 0.1,
                "time_embedding_dim": 16,
                "sampler_type": "ddpm",
                "sample_steps": 10,
                "ddim_eta": 0.0,
                "n_synth_paths": 6,
                "n_real_windows_compare": 6,
                "n_plot_paths": 3,
                "rolling_vol_window": 3,
                "acf_max_lag": 3,
                "hist_bins": 20,
                "normalization_base": 100.0,
                "eval_tail_quantiles": [0.01, 0.05, 0.95, 0.99],
                "eval_exceedance_thresholds": [0.01, 0.02],
                "eval_metrics_version": "v2",
            }

                config_path = os.path.join(tmp, "smoke_diffusion.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)

                validate_config(config)
                run_simulation(run_name="smoke_diffusion", config_path=config_path, config=config)

                scores_dir = os.path.join(output_root, "smoke_diffusion", "csvs", "scores")
                self.assertTrue(os.path.exists(os.path.join(scores_dir, "summary_metrics_train.csv")))
                self.assertTrue(os.path.exists(os.path.join(scores_dir, "summary_metrics_test.csv")))
                self.assertTrue(os.path.exists(os.path.join(scores_dir, "training_loss_history.csv")))
                self.assertTrue(os.path.exists(os.path.join(scores_dir, "noise_schedule.csv")))
                self.assertTrue(os.path.exists(os.path.join(scores_dir, "tail_metrics.csv")))
                self.assertTrue(os.path.exists(os.path.join(scores_dir, "path_risk_metrics.csv")))
        finally:
            if prev_output_root is None:
                os.environ.pop("TIMEGAN_OUTPUT_ROOT", None)
            else:
                os.environ["TIMEGAN_OUTPUT_ROOT"] = prev_output_root


if __name__ == "__main__":
    unittest.main()
