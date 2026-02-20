import numpy as np

from DeepHedging.utils.asian_pricing import (
    arithmetic_control_variate_price_delta_crn,
    arithmetic_control_variate_price_delta_crn_batch,
)


def test_arithmetic_cv_batch_matches_scalar_shared_crn():
    s = np.array([95.0, 100.0, 107.5], dtype=np.float64)
    past_sum = np.array([90.0, 100.0, 115.0], dtype=np.float64)
    past_log_sum = np.log(np.maximum(past_sum, 1e-12))
    fixing_indices = np.arange(22, dtype=np.int32)
    kwargs = dict(
        fixing_indices=fixing_indices,
        current_step=0,
        dt=1.0 / 252.0,
        r=0.05,
        sigma=0.2,
        strike=100.0,
        option_type="call",
        num_simulations=300,
        bump_rel=0.01,
        seed=1234,
    )

    prices_b, deltas_b = arithmetic_control_variate_price_delta_crn_batch(
        S_t=s,
        past_sum=past_sum,
        past_log_sum=past_log_sum,
        seed_mode="shared_crn",
        state_index_offset=0,
        **kwargs,
    )

    prices_s = []
    deltas_s = []
    for i in range(s.shape[0]):
        p_i, d_i = arithmetic_control_variate_price_delta_crn(
            S_t=float(s[i]),
            past_sum=float(past_sum[i]),
            past_log_sum=float(past_log_sum[i]),
            **kwargs,
        )
        prices_s.append(p_i)
        deltas_s.append(d_i)

    np.testing.assert_allclose(prices_b, np.asarray(prices_s, dtype=np.float64), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(deltas_b, np.asarray(deltas_s, dtype=np.float64), rtol=1e-7, atol=1e-7)


def test_arithmetic_cv_batch_per_state_is_reproducible():
    s = np.array([98.0, 101.0, 103.0, 109.0], dtype=np.float64)
    past_sum = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    past_log_sum = np.log(np.maximum(past_sum, 1e-12))
    fixing_indices = np.arange(22, dtype=np.int32)

    kwargs = dict(
        S_t=s,
        past_sum=past_sum,
        past_log_sum=past_log_sum,
        fixing_indices=fixing_indices,
        current_step=1,
        dt=1.0 / 252.0,
        r=0.01,
        sigma=0.15,
        strike=100.0,
        option_type="call",
        num_simulations=250,
        bump_rel=0.01,
        seed=777,
        seed_mode="per_state",
        state_index_offset=0,
    )
    p1, d1 = arithmetic_control_variate_price_delta_crn_batch(**kwargs)
    p2, d2 = arithmetic_control_variate_price_delta_crn_batch(**kwargs)
    np.testing.assert_allclose(p1, p2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(d1, d2, rtol=0.0, atol=0.0)
