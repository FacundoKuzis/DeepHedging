import os
import sys
import unittest

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.utils.lrm_continuation import ContinuationContext, ContinuationValueProvider
from DeepHedging.utils.lrm_engine import compute_lrm_target_batch


class _DummyInstrument:
    def __init__(self, r=0.05, sigma=0.2):
        self.r = float(r)
        self.sigma = float(sigma)


class _LinearProvider(ContinuationValueProvider):
    def estimate_continuation_t1(
        self,
        spot_t1,
        t_index,
        path_prefix=None,
        per_path_r=None,
        per_path_sigma=None,
        seed=None,
    ):
        _ = t_index
        _ = per_path_sigma
        _ = seed
        s_t = np.asarray(path_prefix, dtype=np.float64)[:, -1]
        r_vec = np.asarray(per_path_r, dtype=np.float64).reshape(-1)
        dt = float(self.context.dt)
        d_s = np.asarray(spot_t1, dtype=np.float64) - s_t[:, None] * np.exp(r_vec[:, None] * dt)
        return (1.5 * d_s + 2.0).astype(np.float32)


class TestLrmEngineFormula(unittest.TestCase):
    def test_cov_var_recovers_linear_coefficient(self):
        provider = _LinearProvider()
        provider.prepare(
            ContinuationContext(
                claim=None,
                instrument=_DummyInstrument(r=0.03, sigma=0.2),
                n_steps=22,
                maturity=22 / 252,
                dt=1.0 / 252.0,
            )
        )

        batch = 32
        spot_t = np.full((batch,), 100.0, dtype=np.float64)
        path_prefix = spot_t[:, None]
        r_vec = np.full((batch,), 0.03, dtype=np.float64)
        sigma_vec = np.full((batch,), 0.2, dtype=np.float64)

        hedge = compute_lrm_target_batch(
            spot_t=spot_t,
            t_index=0,
            dt=1.0 / 252.0,
            provider=provider,
            per_path_r=r_vec,
            per_path_sigma=sigma_vec,
            path_prefix=path_prefix,
            outer_paths=256,
            seed=123,
            seed_mode="shared_crn",
        )
        self.assertEqual(hedge.shape, (batch,))
        self.assertTrue(np.isfinite(hedge).all())
        self.assertLess(float(np.mean(np.abs(hedge - 1.5))), 0.05)


if __name__ == "__main__":
    unittest.main()
