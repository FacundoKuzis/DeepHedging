import os
import sys
import unittest

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.utils.timegan_metrics import (
    dependence_metrics_df,
    path_risk_metrics_df,
    tail_metrics_df,
)


class TestTimeganMetricsV2(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        n_paths = 40
        n_steps = 23
        real = np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(n_paths, n_steps - 1)), axis=1))
        synth = np.exp(np.cumsum(rng.normal(0.0001, 0.011, size=(n_paths, n_steps - 1)), axis=1))
        self.real_paths = np.concatenate([np.ones((n_paths, 1)), real], axis=1).astype(np.float32)
        self.synth_paths = np.concatenate([np.ones((n_paths, 1)), synth], axis=1).astype(np.float32)

    def test_dependence_metrics_shapes(self):
        dep = dependence_metrics_df(self.real_paths, self.synth_paths, split="test", max_lag=10, squared=False)
        dep_sq = dependence_metrics_df(self.real_paths, self.synth_paths, split="test", max_lag=10, squared=True)
        self.assertEqual(len(dep), 10)
        self.assertEqual(len(dep_sq), 10)
        self.assertTrue((dep["lag"].to_numpy() == np.arange(1, 11)).all())
        self.assertTrue((dep_sq["lag"].to_numpy() == np.arange(1, 11)).all())

    def test_tail_metrics_contains_quantile_and_exceedance(self):
        tail = tail_metrics_df(
            self.real_paths,
            self.synth_paths,
            split="test",
            quantiles=[0.01, 0.5, 0.99],
            exceedance_thresholds=[0.01, 0.02],
        )
        self.assertIn("quantile_value", set(tail["metric"]))
        self.assertIn("exceedance_prob", set(tail["metric"]))
        self.assertTrue(np.isfinite(tail["abs_error"].to_numpy()).all())

    def test_path_risk_metrics(self):
        risk = path_risk_metrics_df(self.real_paths, self.synth_paths, split="test")
        expected_metrics = {
            "cum_log_return_mean",
            "cum_log_return_std",
            "max_drawdown_mean",
            "max_drawdown_std",
            "max_drawdown_p95",
        }
        self.assertEqual(set(risk["metric"]), expected_metrics)
        self.assertTrue(np.isfinite(risk["abs_error"].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
