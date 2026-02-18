import os
import sys
import unittest

import numpy as np
import tensorflow as tf


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.utils.diffusion_model import build_diffusion_denoiser


class TestDiffusionModelShapes(unittest.TestCase):
    def test_forward_shape_matches_input(self):
        model = build_diffusion_denoiser(
            seq_len=22,
            n_features=2,
            hidden_dim=64,
            num_res_blocks=2,
            dropout=0.1,
            time_embedding_dim=32,
        )
        x = tf.constant(np.random.randn(4, 22, 2), dtype=tf.float32)
        t = tf.constant([0, 3, 10, 21], dtype=tf.int32)
        out = model([x, t], training=False)
        self.assertEqual(tuple(out.shape), (4, 22, 2))
        self.assertTrue(np.isfinite(out.numpy()).all())


if __name__ == "__main__":
    unittest.main()
