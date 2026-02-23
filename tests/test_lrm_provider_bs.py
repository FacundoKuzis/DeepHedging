import os
import sys
import unittest

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.ContingentClaims import EuropeanCall
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.utils.lrm_continuation import ContinuationContext
from DeepHedging.utils.lrm_providers import BSClosedFormContinuationProvider


class TestLrmBsProvider(unittest.TestCase):
    def test_bs_provider_shape_and_finiteness(self):
        ins = GBMStock(S0=100.0, T=22 / 252, N=22, r=0.05, sigma=0.2)
        claim = EuropeanCall(strike=100.0)
        provider = BSClosedFormContinuationProvider()
        provider.prepare(
            ContinuationContext(
                claim=claim,
                instrument=ins,
                n_steps=22,
                maturity=22 / 252,
                dt=1.0 / 252.0,
                strike=100.0,
                option_type="call",
            )
        )

        spot_t1 = np.array([[95.0, 100.0, 105.0], [90.0, 100.0, 110.0]], dtype=np.float64)
        out = provider.estimate_continuation_t1(
            spot_t1=spot_t1,
            t_index=0,
            per_path_r=np.array([0.05, 0.03], dtype=np.float64),
            per_path_sigma=np.array([0.2, 0.25], dtype=np.float64),
        )
        self.assertEqual(out.shape, spot_t1.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
