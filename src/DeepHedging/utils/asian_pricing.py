import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def resolve_fixing_indices(fixing_indices, n_steps):
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if fixing_indices is None:
        return np.arange(n_steps, dtype=np.int32)
    arr = np.array(fixing_indices, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        raise ValueError("fixing_indices cannot be empty.")
    if np.any(arr < 0) or np.any(arr >= n_steps):
        raise ValueError(f"fixing_indices must be in [0, {n_steps - 1}]. Got {arr}.")
    arr = np.unique(arr)
    arr.sort()
    return arr


def build_running_asian_state(paths_2d, fixing_indices=None):
    """
    Args:
        paths_2d: tf.Tensor shape (batch, time), strictly positive prices.
    Returns:
        running_arith: (batch, time)
        running_geo: (batch, time)
    """
    n_steps = int(paths_2d.shape[1])
    fixings = resolve_fixing_indices(fixing_indices, n_steps)
    is_fix_np = np.zeros((n_steps,), dtype=np.float32)
    is_fix_np[fixings] = 1.0
    is_fix = tf.constant(is_fix_np, dtype=tf.float32)
    is_fix_2d = tf.reshape(is_fix, (1, -1))

    safe_paths = tf.maximum(paths_2d, 1e-12)
    cum_sum = tf.cumsum(paths_2d * is_fix_2d, axis=1)
    cum_log_sum = tf.cumsum(tf.math.log(safe_paths) * is_fix_2d, axis=1)

    counts = tf.cumsum(is_fix)
    counts_safe = tf.where(counts > 0.0, counts, tf.ones_like(counts))
    counts_2d = tf.reshape(counts_safe, (1, -1))
    has_fix_2d = tf.reshape(counts > 0.0, (1, -1))

    running_arith = cum_sum / counts_2d
    running_geo = tf.exp(cum_log_sum / counts_2d)

    # Before first fixing, fall back to spot so features remain meaningful.
    running_arith = tf.where(has_fix_2d, running_arith, paths_2d)
    running_geo = tf.where(has_fix_2d, running_geo, paths_2d)
    return running_arith, running_geo


def _future_fixing_steps(current_step, fixing_indices):
    return np.array([idx - current_step for idx in fixing_indices if idx > current_step], dtype=np.int32)


def _sum_min_times(times):
    if times.size == 0:
        return 0.0
    return float(np.minimum.outer(times, times).sum())


def geometric_conditional_price_tf(
    S,
    past_log_sum,
    total_fixings,
    future_fixing_steps,
    dt,
    r,
    sigma,
    strike,
    option_type,
):
    """
    Closed-form price for a discrete geometric-asian option conditional on past fixings.
    All discounting is done from current time to maturity (last future fixing).
    """
    S = tf.convert_to_tensor(S, dtype=tf.float32)
    past_log_sum = tf.convert_to_tensor(past_log_sum, dtype=tf.float32)
    strike_t = tf.constant(float(strike), dtype=tf.float32)
    option_type = option_type.lower()

    n_future = int(len(future_fixing_steps))
    if n_future == 0:
        deterministic_geo = tf.exp(past_log_sum / float(total_fixings))
        if option_type == "call":
            payoff = tf.maximum(deterministic_geo - strike_t, 0.0)
        elif option_type == "put":
            payoff = tf.maximum(strike_t - deterministic_geo, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")
        return payoff

    tau = future_fixing_steps.astype(np.float64) * float(dt)
    sum_tau = float(np.sum(tau))
    sum_min_tau = _sum_min_times(tau)
    total_fixings_f = float(total_fixings)
    n_future_f = float(n_future)

    drift = (float(r) - 0.5 * float(sigma) ** 2)
    mu = (n_future_f / total_fixings_f) * tf.math.log(tf.maximum(S, 1e-12))
    mu = mu + drift * (sum_tau / total_fixings_f)
    var = (float(sigma) ** 2) * (sum_min_tau / (total_fixings_f**2))
    var_t = tf.constant(max(var, 1e-12), dtype=tf.float32)
    std_t = tf.sqrt(var_t)

    m = tf.math.log(tf.maximum(tf.exp(past_log_sum / total_fixings_f), 1e-12)) + mu
    log_k = tf.math.log(strike_t)
    d1 = (m - log_k + var_t) / std_t
    d2 = (m - log_k) / std_t

    normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
    exp_term = tf.exp(m + 0.5 * var_t)
    call_no_disc = exp_term * normal.cdf(d1) - strike_t * normal.cdf(d2)
    put_no_disc = strike_t * normal.cdf(-d2) - exp_term * normal.cdf(-d1)

    t_rem = float(np.max(tau))
    discount = tf.constant(np.exp(-float(r) * t_rem), dtype=tf.float32)
    if option_type == "call":
        return discount * call_no_disc
    if option_type == "put":
        return discount * put_no_disc
    raise ValueError("option_type must be 'call' or 'put'.")


def geometric_conditional_delta_bump_tf(
    S,
    past_log_sum,
    total_fixings,
    future_fixing_steps,
    dt,
    r,
    sigma,
    strike,
    option_type,
    bump_rel=0.01,
):
    S = tf.convert_to_tensor(S, dtype=tf.float32)
    eps = tf.maximum(tf.abs(S) * float(bump_rel), tf.constant(1e-6, dtype=tf.float32))
    price_up = geometric_conditional_price_tf(
        S + eps,
        past_log_sum,
        total_fixings,
        future_fixing_steps,
        dt,
        r,
        sigma,
        strike,
        option_type,
    )
    price_down = geometric_conditional_price_tf(
        tf.maximum(S - eps, 1e-8),
        past_log_sum,
        total_fixings,
        future_fixing_steps,
        dt,
        r,
        sigma,
        strike,
        option_type,
    )
    return (price_up - price_down) / (2.0 * eps)


def _simulate_future_fixings_from_normals(S0, future_steps, dt, r, sigma, normals):
    if len(future_steps) == 0:
        return np.empty((normals.shape[0], 0), dtype=np.float64)
    max_step = int(np.max(future_steps))
    z = normals[:, :max_step]
    drift = (float(r) - 0.5 * float(sigma) ** 2) * float(dt)
    vol = float(sigma) * np.sqrt(float(dt))
    log_increments = drift + vol * z
    log_paths = np.log(max(S0, 1e-12)) + np.cumsum(log_increments, axis=1)
    step_idx = np.array(future_steps, dtype=np.int32) - 1
    return np.exp(log_paths[:, step_idx])


def _normal_cdf_approx(x):
    """
    Fast normal CDF approximation (Abramowitz-Stegun style), vectorized in NumPy.
    Avoids TensorFlow/TFP overhead inside tight Monte Carlo loops.
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    erf_approx = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-z * z)
    erf_approx = sign * erf_approx
    return 0.5 * (1.0 + erf_approx)


def _geometric_conditional_price_np_batch(
    S,
    past_log_sum,
    total_fixings,
    future_fixing_steps,
    dt,
    r,
    sigma,
    strike,
    option_type,
):
    """
    NumPy batch equivalent of geometric_conditional_price_tf.
    Args:
        S: np.ndarray shape (batch,)
        past_log_sum: np.ndarray shape (batch,)
    Returns:
        np.ndarray shape (batch,)
    """
    S = np.asarray(S, dtype=np.float64).reshape(-1)
    past_log_sum = np.asarray(past_log_sum, dtype=np.float64).reshape(-1)
    if S.shape != past_log_sum.shape:
        raise ValueError("S and past_log_sum must have the same shape.")

    option_type = str(option_type).lower()
    n_future = int(len(future_fixing_steps))
    total_fixings_f = float(total_fixings)
    strike_f = float(strike)

    if n_future == 0:
        deterministic_geo = np.exp(past_log_sum / total_fixings_f)
        if option_type == "call":
            return np.maximum(deterministic_geo - strike_f, 0.0)
        if option_type == "put":
            return np.maximum(strike_f - deterministic_geo, 0.0)
        raise ValueError("option_type must be 'call' or 'put'.")

    tau = np.asarray(future_fixing_steps, dtype=np.float64) * float(dt)
    sum_tau = float(np.sum(tau))
    sum_min_tau = _sum_min_times(tau)
    n_future_f = float(n_future)

    drift = float(r) - 0.5 * float(sigma) ** 2
    mu = (n_future_f / total_fixings_f) * np.log(np.maximum(S, 1e-12))
    mu = mu + drift * (sum_tau / total_fixings_f)
    var = (float(sigma) ** 2) * (sum_min_tau / (total_fixings_f**2))
    var = max(var, 1e-12)
    std = np.sqrt(var)

    m = (past_log_sum / total_fixings_f) + mu
    log_k = np.log(strike_f)
    d1 = (m - log_k + var) / std
    d2 = (m - log_k) / std

    exp_term = np.exp(m + 0.5 * var)
    nd1 = _normal_cdf_approx(d1)
    nd2 = _normal_cdf_approx(d2)

    call_no_disc = exp_term * nd1 - strike_f * nd2
    put_no_disc = strike_f * _normal_cdf_approx(-d2) - exp_term * _normal_cdf_approx(-d1)

    t_rem = float(np.max(tau))
    discount = np.exp(-float(r) * t_rem)
    if option_type == "call":
        return discount * call_no_disc
    if option_type == "put":
        return discount * put_no_disc
    raise ValueError("option_type must be 'call' or 'put'.")


def _prepare_shared_future_factors(future_steps, dt, r, sigma, normals):
    """
    Build reusable factors for S-multiplicative future prices under GBM.
    Returns:
        sum_factors: shape (num_simulations,)
        sum_log_factors: shape (num_simulations,)
    """
    if len(future_steps) == 0:
        return np.empty((normals.shape[0],), dtype=np.float64), np.empty((normals.shape[0],), dtype=np.float64)

    max_step = int(np.max(future_steps))
    z = normals[:, :max_step]
    drift = (float(r) - 0.5 * float(sigma) ** 2) * float(dt)
    vol = float(sigma) * np.sqrt(float(dt))
    log_increments = drift + vol * z
    log_factors_path = np.cumsum(log_increments, axis=1)
    step_idx = np.asarray(future_steps, dtype=np.int32) - 1
    selected_log = log_factors_path[:, step_idx]  # (num_simulations, n_future_fixings)
    selected = np.exp(selected_log)
    sum_factors = np.sum(selected, axis=1)
    sum_log_factors = np.sum(selected_log, axis=1)
    return sum_factors, sum_log_factors


def _arithmetic_cv_adjusted_price_from_shared_factors(
    spot_vec,
    past_sum_vec,
    past_log_sum_vec,
    total_fixings,
    n_future,
    sum_factors,
    sum_log_factors,
    r,
    sigma,
    dt,
    future_steps,
    strike,
    option_type,
):
    """
    Vectorized adjusted arithmetic-asian price for a batch of states sharing MC factors.
    """
    spot_vec = np.asarray(spot_vec, dtype=np.float64).reshape(-1)
    past_sum_vec = np.asarray(past_sum_vec, dtype=np.float64).reshape(-1)
    past_log_sum_vec = np.asarray(past_log_sum_vec, dtype=np.float64).reshape(-1)
    if not (spot_vec.shape == past_sum_vec.shape == past_log_sum_vec.shape):
        raise ValueError("spot_vec, past_sum_vec and past_log_sum_vec must have the same shape.")

    m = int(sum_factors.shape[0])
    if m <= 1:
        raise ValueError("num_simulations must be > 1 for stable control variate covariance.")

    total_fixings_f = float(total_fixings)
    disc = np.exp(-float(r) * float(np.max(future_steps) * dt))

    spot_col = spot_vec[:, None]
    past_sum_col = past_sum_vec[:, None]
    past_log_col = past_log_sum_vec[:, None]

    sum_factors_row = sum_factors[None, :]
    sum_log_factors_row = sum_log_factors[None, :]

    arith_avg = (past_sum_col + spot_col * sum_factors_row) / total_fixings_f
    geo_log = (
        past_log_col
        + (float(n_future) * np.log(np.maximum(spot_col, 1e-12)))
        + sum_log_factors_row
    ) / total_fixings_f
    geo_avg = np.exp(geo_log)

    option_type = str(option_type).lower()
    strike_f = float(strike)
    if option_type == "call":
        arith_payoff = np.maximum(arith_avg - strike_f, 0.0)
        geo_payoff = np.maximum(geo_avg - strike_f, 0.0)
    elif option_type == "put":
        arith_payoff = np.maximum(strike_f - arith_avg, 0.0)
        geo_payoff = np.maximum(strike_f - geo_avg, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    arith_disc = disc * arith_payoff
    geo_disc = disc * geo_payoff

    geo_analytic = _geometric_conditional_price_np_batch(
        S=spot_vec,
        past_log_sum=past_log_sum_vec,
        total_fixings=total_fixings,
        future_fixing_steps=future_steps,
        dt=dt,
        r=r,
        sigma=sigma,
        strike=strike,
        option_type=option_type,
    )

    arith_mean = np.mean(arith_disc, axis=1)
    geo_mean = np.mean(geo_disc, axis=1)

    arith_centered = arith_disc - arith_mean[:, None]
    geo_centered = geo_disc - geo_mean[:, None]
    cov = np.sum(arith_centered * geo_centered, axis=1) / float(m - 1)
    var = np.sum(geo_centered * geo_centered, axis=1) / float(m - 1)
    beta = np.zeros_like(cov)
    np.divide(cov, var, out=beta, where=var > 1e-12)

    adjusted = arith_mean - beta * (geo_mean - geo_analytic)
    return adjusted


def arithmetic_control_variate_price_delta_crn_batch(
    S_t,
    past_sum,
    past_log_sum,
    fixing_indices,
    current_step,
    dt,
    r,
    sigma,
    strike,
    option_type,
    num_simulations=10_000,
    bump_rel=0.01,
    seed=33,
    seed_mode="shared_crn",
    state_index_offset=0,
):
    """
    Batch version of conditional arithmetic-asian pricing with geometric control variate and CRN delta.
    Returns:
        prices: np.ndarray (batch,)
        deltas: np.ndarray (batch,)
    """
    s_arr = np.asarray(S_t, dtype=np.float64).reshape(-1)
    past_sum_arr = np.asarray(past_sum, dtype=np.float64).reshape(-1)
    past_log_sum_arr = np.asarray(past_log_sum, dtype=np.float64).reshape(-1)
    if not (s_arr.shape == past_sum_arr.shape == past_log_sum_arr.shape):
        raise ValueError("S_t, past_sum and past_log_sum must have same shape.")
    if int(num_simulations) <= 1:
        raise ValueError("num_simulations must be > 1.")

    fixings = np.asarray(fixing_indices, dtype=np.int32).reshape(-1)
    total_fixings = int(len(fixings))
    if total_fixings <= 0:
        raise ValueError("fixing_indices must contain at least one fixing.")

    future_steps = _future_fixing_steps(int(current_step), fixings)
    option_type = str(option_type).lower()

    if len(future_steps) == 0:
        avg = past_sum_arr / float(total_fixings)
        if option_type == "call":
            prices = np.maximum(avg - float(strike), 0.0)
        elif option_type == "put":
            prices = np.maximum(float(strike) - avg, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")
        deltas = np.zeros_like(prices)
        return prices.astype(np.float64), deltas.astype(np.float64)

    seed_mode = str(seed_mode).strip().lower()
    if seed_mode not in {"shared_crn", "per_state"}:
        raise ValueError("seed_mode must be 'shared_crn' or 'per_state'.")

    eps = np.maximum(np.abs(s_arr) * float(bump_rel), 1e-6)
    n_future = len(future_steps)

    if seed_mode == "shared_crn":
        rng = np.random.default_rng(int(seed))
        max_step = int(np.max(future_steps))
        normals = rng.normal(0.0, 1.0, size=(int(num_simulations), max_step))
        sum_factors, sum_log_factors = _prepare_shared_future_factors(
            future_steps=future_steps,
            dt=dt,
            r=r,
            sigma=sigma,
            normals=normals,
        )
        price_mid = _arithmetic_cv_adjusted_price_from_shared_factors(
            spot_vec=s_arr,
            past_sum_vec=past_sum_arr,
            past_log_sum_vec=past_log_sum_arr,
            total_fixings=total_fixings,
            n_future=n_future,
            sum_factors=sum_factors,
            sum_log_factors=sum_log_factors,
            r=r,
            sigma=sigma,
            dt=dt,
            future_steps=future_steps,
            strike=strike,
            option_type=option_type,
        )
        price_up = _arithmetic_cv_adjusted_price_from_shared_factors(
            spot_vec=s_arr + eps,
            past_sum_vec=past_sum_arr,
            past_log_sum_vec=past_log_sum_arr,
            total_fixings=total_fixings,
            n_future=n_future,
            sum_factors=sum_factors,
            sum_log_factors=sum_log_factors,
            r=r,
            sigma=sigma,
            dt=dt,
            future_steps=future_steps,
            strike=strike,
            option_type=option_type,
        )
        price_down = _arithmetic_cv_adjusted_price_from_shared_factors(
            spot_vec=np.maximum(s_arr - eps, 1e-8),
            past_sum_vec=past_sum_arr,
            past_log_sum_vec=past_log_sum_arr,
            total_fixings=total_fixings,
            n_future=n_future,
            sum_factors=sum_factors,
            sum_log_factors=sum_log_factors,
            r=r,
            sigma=sigma,
            dt=dt,
            future_steps=future_steps,
            strike=strike,
            option_type=option_type,
        )
        deltas = (price_up - price_down) / (2.0 * eps)
        return price_mid.astype(np.float64), deltas.astype(np.float64)

    prices = np.zeros_like(s_arr, dtype=np.float64)
    deltas = np.zeros_like(s_arr, dtype=np.float64)
    for i in range(s_arr.shape[0]):
        state_seed = int(seed) + int(state_index_offset) + i
        rng = np.random.default_rng(state_seed)
        max_step = int(np.max(future_steps))
        normals = rng.normal(0.0, 1.0, size=(int(num_simulations), max_step))
        sum_factors, sum_log_factors = _prepare_shared_future_factors(
            future_steps=future_steps,
            dt=dt,
            r=r,
            sigma=sigma,
            normals=normals,
        )
        spot_i = np.asarray([s_arr[i]], dtype=np.float64)
        ps_i = np.asarray([past_sum_arr[i]], dtype=np.float64)
        pls_i = np.asarray([past_log_sum_arr[i]], dtype=np.float64)
        eps_i = float(eps[i])

        mid_i = _arithmetic_cv_adjusted_price_from_shared_factors(
            spot_vec=spot_i,
            past_sum_vec=ps_i,
            past_log_sum_vec=pls_i,
            total_fixings=total_fixings,
            n_future=n_future,
            sum_factors=sum_factors,
            sum_log_factors=sum_log_factors,
            r=r,
            sigma=sigma,
            dt=dt,
            future_steps=future_steps,
            strike=strike,
            option_type=option_type,
        )[0]
        up_i = _arithmetic_cv_adjusted_price_from_shared_factors(
            spot_vec=spot_i + eps_i,
            past_sum_vec=ps_i,
            past_log_sum_vec=pls_i,
            total_fixings=total_fixings,
            n_future=n_future,
            sum_factors=sum_factors,
            sum_log_factors=sum_log_factors,
            r=r,
            sigma=sigma,
            dt=dt,
            future_steps=future_steps,
            strike=strike,
            option_type=option_type,
        )[0]
        down_i = _arithmetic_cv_adjusted_price_from_shared_factors(
            spot_vec=np.maximum(spot_i - eps_i, 1e-8),
            past_sum_vec=ps_i,
            past_log_sum_vec=pls_i,
            total_fixings=total_fixings,
            n_future=n_future,
            sum_factors=sum_factors,
            sum_log_factors=sum_log_factors,
            r=r,
            sigma=sigma,
            dt=dt,
            future_steps=future_steps,
            strike=strike,
            option_type=option_type,
        )[0]
        prices[i] = mid_i
        deltas[i] = (up_i - down_i) / (2.0 * eps_i)
    return prices, deltas


def arithmetic_control_variate_price_delta_crn_batch_worker(kwargs):
    """
    Process-safe worker wrapper for parallel execution.
    """
    return arithmetic_control_variate_price_delta_crn_batch(**kwargs)


def arithmetic_control_variate_price_delta_crn(
    S_t,
    past_sum,
    past_log_sum,
    fixing_indices,
    current_step,
    dt,
    r,
    sigma,
    strike,
    option_type,
    num_simulations=10_000,
    bump_rel=0.01,
    seed=33,
):
    """
    Conditional arithmetic-asian pricing with geometric control variate and CRN delta.
    """
    prices, deltas = arithmetic_control_variate_price_delta_crn_batch(
        S_t=np.asarray([S_t], dtype=np.float64),
        past_sum=np.asarray([past_sum], dtype=np.float64),
        past_log_sum=np.asarray([past_log_sum], dtype=np.float64),
        fixing_indices=fixing_indices,
        current_step=current_step,
        dt=dt,
        r=r,
        sigma=sigma,
        strike=strike,
        option_type=option_type,
        num_simulations=num_simulations,
        bump_rel=bump_rel,
        seed=seed,
        seed_mode="shared_crn",
        state_index_offset=0,
    )
    return float(prices[0]), float(deltas[0])
