import os
import sys
import unittest

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.utils.timegan_transforms import fit_transform_1d, inverse_transform_1d


class TestTimeganTransforms(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.values = rng.normal(loc=0.0, scale=0.02, size=500).astype(np.float32)

    def _roundtrip(self, method: str, atol: float = 1e-3) -> None:
        transformed, state = fit_transform_1d(self.values, method=method, eps=1e-6)
        restored = inverse_transform_1d(transformed, state=state, eps=1e-6)
        self.assertEqual(transformed.shape, self.values.shape)
        self.assertEqual(restored.shape, self.values.shape)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.isfinite(restored).all())
        np.testing.assert_allclose(restored, self.values, atol=atol, rtol=0)

    def test_minmax_roundtrip(self):
        self._roundtrip("minmax", atol=1e-6)

    def test_gaussian_cdf_roundtrip(self):
        self._roundtrip("gaussian_cdf", atol=5e-4)

    def test_empirical_cdf_roundtrip(self):
        self._roundtrip("empirical_cdf", atol=5e-3)

    def test_eps_clips_u_space(self):
        transformed, _ = fit_transform_1d(self.values, method="gaussian_cdf", eps=1e-3)
        self.assertGreaterEqual(float(np.min(transformed)), 1e-3 - 1e-12)
        self.assertLessEqual(float(np.max(transformed)), 1 - 1e-3 + 1e-12)


if __name__ == "__main__":
    unittest.main()
