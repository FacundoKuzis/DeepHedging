import numpy as np
import tensorflow as tf

from DeepHedging.Agents import (
    ArithmeticAsianControlVariateAgent,
    DeltaHedgingAgent,
    GeometricAsianDeltaHedgingAgent,
)
from DeepHedging.ContingentClaims import (
    AsianArithmeticCall,
    AsianGeometricCall,
    EuropeanCall,
)
from DeepHedging.HedgingInstruments import GBMStock


def _t_minus_t(batch_size, n_steps, dt):
    single = tf.range(n_steps, 0, -1, dtype=tf.float32) * tf.constant(dt, dtype=tf.float32)
    return tf.tile(tf.expand_dims(single, axis=0), [batch_size, 1])


def test_claim_fixing_calendar_applies_correctly():
    paths = tf.constant([[100.0, 110.0, 120.0, 130.0, 140.0]], dtype=tf.float32)
    claim = AsianArithmeticCall(strike=120.0, fixing_indices=[0, 2, 4])
    payoff = claim.calculate_payoff(paths).numpy()
    expected_avg = (100.0 + 120.0 + 140.0) / 3.0
    expected = max(expected_avg - 120.0, 0.0)
    assert np.isclose(payoff[0], expected, atol=1e-8)


def test_no_trade_band_reduces_rebalancing():
    stock = GBMStock(S0=100.0, T=2 / 252, N=2, r=0.0, sigma=0.2)
    claim = EuropeanCall(strike=100.0)
    paths = tf.constant([[[100.0], [100.0], [100.0]]], dtype=tf.float32)
    t_minus_t = _t_minus_t(batch_size=1, n_steps=2, dt=stock.dt)

    free_agent = DeltaHedgingAgent(stock, claim, no_trade_band=0.0)
    band_agent = DeltaHedgingAgent(stock, claim, no_trade_band=1.0)

    actions_free = free_agent.process_batch(paths, t_minus_t).numpy()
    actions_band = band_agent.process_batch(paths, t_minus_t).numpy()

    # Terminal action is always zero.
    assert np.isclose(actions_band[0, -1, 0], 0.0, atol=1e-8)
    # With a wide no-trade band, second rebalance should be shut down.
    assert np.isclose(actions_band[0, 1, 0], 0.0, atol=1e-8)
    # Free agent should differ from band-limited execution.
    assert not np.allclose(actions_free, actions_band)


def test_geometric_asian_agent_stateful_delta_shapes():
    stock = GBMStock(S0=100.0, T=4 / 252, N=4, r=0.01, sigma=0.2)
    claim = AsianGeometricCall(strike=100.0, fixing_indices=[0, 2, 4])
    agent = GeometricAsianDeltaHedgingAgent(stock, claim, bump_size=0.01, no_trade_band=0.0)

    paths = tf.expand_dims(stock.generate_paths(num_paths=3, random_seed=123), axis=-1)
    t_minus_t = _t_minus_t(batch_size=3, n_steps=stock.N, dt=stock.dt)
    actions = agent.process_batch(paths, t_minus_t)

    assert actions.shape == (3, stock.N + 1, 1)
    assert np.isfinite(actions.numpy()).all()
    assert np.isfinite(float(agent.get_model_price()))


def test_arithmetic_control_variate_agent_stateful_delta_shapes():
    stock = GBMStock(S0=100.0, T=4 / 252, N=4, r=0.01, sigma=0.2)
    claim = AsianArithmeticCall(strike=100.0, fixing_indices=[0, 2, 4])
    agent = ArithmeticAsianControlVariateAgent(
        stock_model=stock,
        option_class=claim,
        num_simulations=200,
        bump_size=0.01,
        seed=11,
        no_trade_band=0.0,
    )

    paths = tf.expand_dims(stock.generate_paths(num_paths=2, random_seed=999), axis=-1)
    t_minus_t = _t_minus_t(batch_size=2, n_steps=stock.N, dt=stock.dt)
    actions = agent.process_batch(paths, t_minus_t)

    assert actions.shape == (2, stock.N + 1, 1)
    assert np.isfinite(actions.numpy()).all()
    assert np.isfinite(float(agent.get_model_price()))

