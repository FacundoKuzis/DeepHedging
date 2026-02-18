import os
import sys
import unittest

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.utils.diffusion_schedule import build_diffusion_schedule


class TestDiffusionSchedule(unittest.TestCase):
    def test_linear_schedule_monotonicity(self):
        sched = build_diffusion_schedule(
            timesteps=100,
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=2e-2,
        )
        self.assertEqual(sched.timesteps, 100)
        self.assertEqual(len(sched.betas), 100)
        self.assertTrue(np.all(np.diff(sched.betas) >= 0))
        self.assertTrue(np.all(sched.alphas_cumprod > 0))
        self.assertTrue(np.all(np.diff(sched.alphas_cumprod) <= 0))

    def test_cosine_schedule_shapes(self):
        sched = build_diffusion_schedule(
            timesteps=64,
            beta_schedule="cosine",
            beta_start=1e-4,
            beta_end=2e-2,
        )
        self.assertEqual(sched.betas.shape, (64,))
        self.assertEqual(sched.posterior_variance.shape, (64,))
        self.assertTrue(np.isfinite(sched.posterior_mean_coef1).all())
        self.assertTrue(np.isfinite(sched.posterior_mean_coef2).all())


if __name__ == "__main__":
    unittest.main()
