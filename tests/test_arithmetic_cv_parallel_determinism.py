import numpy as np
import tensorflow as tf

from DeepHedging.Agents import ArithmeticAsianControlVariateAgent
from DeepHedging.ContingentClaims import AsianArithmeticCall
from DeepHedging.HedgingInstruments import GBMStock


def _t_minus_t(batch_size, n_steps, dt):
    single = tf.range(n_steps, 0, -1, dtype=tf.float32) * tf.constant(dt, dtype=tf.float32)
    return tf.tile(tf.expand_dims(single, axis=0), [batch_size, 1])


def test_arithmetic_cv_parallel_matches_sequential_thread_backend():
    stock = GBMStock(S0=100.0, T=4 / 252, N=4, r=0.01, sigma=0.2)
    claim = AsianArithmeticCall(strike=100.0, fixing_indices=[0, 1, 2, 3, 4])
    paths = tf.expand_dims(stock.generate_paths(num_paths=6, random_seed=123), axis=-1)
    t_minus_t = _t_minus_t(batch_size=6, n_steps=stock.N, dt=stock.dt)

    seq_agent = ArithmeticAsianControlVariateAgent(
        stock_model=stock,
        option_class=claim,
        num_simulations=200,
        bump_size=0.01,
        seed=21,
        mc_state_chunk_size=2,
        mc_seed_mode="shared_crn",
        parallel_enabled=False,
    )
    par_agent = ArithmeticAsianControlVariateAgent(
        stock_model=stock,
        option_class=claim,
        num_simulations=200,
        bump_size=0.01,
        seed=21,
        mc_state_chunk_size=2,
        mc_seed_mode="shared_crn",
        parallel_enabled=True,
        n_workers=2,
        parallel_backend="thread",
        parallel_chunk_size=2,
        parallel_min_states=1,
    )

    actions_seq = seq_agent.process_batch(paths, t_minus_t).numpy()
    actions_par = par_agent.process_batch(paths, t_minus_t).numpy()

    assert actions_seq.shape == actions_par.shape
    np.testing.assert_allclose(actions_seq, actions_par, rtol=0.0, atol=1e-8)
