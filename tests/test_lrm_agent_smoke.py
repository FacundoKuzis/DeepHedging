import os
import sys
import unittest

import numpy as np
import tensorflow as tf


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.Agents import LocalRiskMinimizationAgent
from DeepHedging.ContingentClaims import AsianArithmeticCall, EuropeanCall
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.utils.lrm_continuation import build_continuation_provider


class TestLrmAgentSmoke(unittest.TestCase):
    def setUp(self):
        self.stock = GBMStock(S0=100.0, T=22 / 252, N=22, r=0.05, sigma=0.2)
        self.claim = EuropeanCall(strike=100.0)
        self.paths_2d = self.stock.generate_paths(64, random_seed=7)
        self.paths_3d = tf.expand_dims(self.paths_2d, axis=-1)
        self.t_minus_t = tf.tile(
            tf.expand_dims(tf.range(22, 0, -1, dtype=tf.float32) * float(self.stock.dt), axis=0),
            [64, 1],
        )

    def test_factory_filters_irrelevant_kwargs(self):
        provider = build_continuation_provider(
            "bs_closed_form",
            inner_paths=999,  # not part of BS ctor
            random_unused_arg=True,  # not part of BS ctor
        )
        self.assertEqual(getattr(provider, "provider_name", None), "bs_closed_form")

    def test_bs_provider_actions_shape_with_rank2_input(self):
        agent = LocalRiskMinimizationAgent(
            self.stock,
            self.claim,
            lrm_provider="bs_closed_form",
            lrm_outer_paths=32,
        )
        actions = agent.process_batch(self.paths_2d, self.t_minus_t)
        self.assertEqual(tuple(actions.shape), (64, 23, 1))
        self.assertTrue(np.isfinite(actions.numpy()).all())

    def test_mc_and_lsm_provider_smoke(self):
        for provider_name, kwargs in [
            ("monte_carlo", {"lrm_mc_inner_paths": 64}),
            ("lsm", {"lrm_lsm_train_paths": 2000}),
        ]:
            agent = LocalRiskMinimizationAgent(
                self.stock,
                self.claim,
                lrm_provider=provider_name,
                lrm_outer_paths=16,
                **kwargs,
            )
            actions = agent.process_batch(self.paths_3d, self.t_minus_t)
            prices = agent.get_model_price_batch(
                path_s0=np.full((64,), 100.0, dtype=np.float32),
                path_r=np.full((64,), 0.05, dtype=np.float32),
                path_sigma=np.full((64,), 0.2, dtype=np.float32),
            )
            self.assertEqual(tuple(actions.shape), (64, 23, 1))
            self.assertEqual(tuple(prices.shape), (64,))
            self.assertTrue(np.isfinite(actions.numpy()).all())
            self.assertTrue(np.isfinite(prices.numpy()).all())

    def test_asian_mc_and_lsm_provider_smoke(self):
        claim = AsianArithmeticCall(strike=100.0, fixing_indices=list(range(1, 23)))
        batch = 16
        paths_2d = self.stock.generate_paths(batch, random_seed=13)
        paths_3d = tf.expand_dims(paths_2d, axis=-1)
        t_minus_t = tf.tile(
            tf.expand_dims(tf.range(22, 0, -1, dtype=tf.float32) * float(self.stock.dt), axis=0),
            [batch, 1],
        )
        for provider_name, kwargs in [
            ("asian_monte_carlo", {"lrm_mc_inner_paths": 32}),
            ("asian_lsmc", {"lrm_lsm_train_paths": 2000}),
        ]:
            agent = LocalRiskMinimizationAgent(
                self.stock,
                claim,
                lrm_provider=provider_name,
                lrm_outer_paths=8,
                **kwargs,
            )
            actions = agent.process_batch(paths_3d, t_minus_t)
            prices = agent.get_model_price_batch(
                path_s0=np.full((batch,), 100.0, dtype=np.float32),
                path_r=np.full((batch,), 0.05, dtype=np.float32),
                path_sigma=np.full((batch,), 0.2, dtype=np.float32),
            )
            self.assertEqual(tuple(actions.shape), (batch, 23, 1))
            self.assertEqual(tuple(prices.shape), (batch,))
            self.assertTrue(np.isfinite(actions.numpy()).all())
            self.assertTrue(np.isfinite(prices.numpy()).all())


if __name__ == "__main__":
    unittest.main()
