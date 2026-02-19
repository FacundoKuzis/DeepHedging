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
    fixings = np.array(fixing_indices, dtype=np.int32)
    total_fixings = int(len(fixings))
    future_steps = _future_fixing_steps(int(current_step), fixings)
    option_type = option_type.lower()

    if len(future_steps) == 0:
        avg = past_sum / float(total_fixings)
        if option_type == "call":
            payoff = max(avg - strike, 0.0)
        elif option_type == "put":
            payoff = max(strike - avg, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")
        return float(payoff), 0.0

    rng = np.random.default_rng(int(seed))
    max_step = int(np.max(future_steps))
    normals = rng.normal(0.0, 1.0, size=(int(num_simulations), max_step))
    eps = max(abs(float(S_t)) * float(bump_rel), 1e-6)

    def _price_for_spot(spot):
        fut = _simulate_future_fixings_from_normals(
            S0=spot,
            future_steps=future_steps,
            dt=dt,
            r=r,
            sigma=sigma,
            normals=normals,
        )
        arith_avg = (past_sum + fut.sum(axis=1)) / float(total_fixings)
        geo_log = (past_log_sum + np.log(np.maximum(fut, 1e-12)).sum(axis=1)) / float(total_fixings)
        geo_avg = np.exp(geo_log)

        if option_type == "call":
            arith_payoff = np.maximum(arith_avg - strike, 0.0)
            geo_payoff = np.maximum(geo_avg - strike, 0.0)
        elif option_type == "put":
            arith_payoff = np.maximum(strike - arith_avg, 0.0)
            geo_payoff = np.maximum(strike - geo_avg, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        t_rem = float(np.max(future_steps) * dt)
        disc = np.exp(-float(r) * t_rem)
        arith_disc = disc * arith_payoff
        geo_disc = disc * geo_payoff
        geo_analytic = float(
            geometric_conditional_price_tf(
                tf.constant([spot], dtype=tf.float32),
                tf.constant([past_log_sum], dtype=tf.float32),
                total_fixings=total_fixings,
                future_fixing_steps=future_steps,
                dt=dt,
                r=r,
                sigma=sigma,
                strike=strike,
                option_type=option_type,
            ).numpy()[0]
        )
        geo_var = float(np.var(geo_disc, ddof=1)) if geo_disc.size > 1 else 0.0
        if geo_var > 1e-12 and geo_disc.size > 1:
            cov = float(np.cov(arith_disc, geo_disc, ddof=1)[0, 1])
            beta = cov / geo_var
        else:
            beta = 0.0

        adjusted = np.mean(arith_disc - beta * (geo_disc - geo_analytic))
        return float(adjusted)

    price_mid = _price_for_spot(float(S_t))
    price_up = _price_for_spot(float(S_t) + eps)
    price_down = _price_for_spot(max(float(S_t) - eps, 1e-8))
    delta = (price_up - price_down) / (2.0 * eps)
    return float(price_mid), float(delta)
