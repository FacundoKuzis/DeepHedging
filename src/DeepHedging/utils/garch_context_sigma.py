import numpy as np


def _clip_sigma(sigma, sigma_floor, sigma_cap):
    out = np.maximum(np.asarray(sigma, dtype=np.float64), float(sigma_floor))
    if sigma_cap is not None:
        out = np.minimum(out, float(sigma_cap))
    return out


def _clip_df(df_values, df_floor, df_cap):
    out = np.maximum(np.asarray(df_values, dtype=np.float64), float(df_floor))
    if df_cap is not None:
        out = np.minimum(out, float(df_cap))
    return out


def _as_path_vector(value, n_paths: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((int(n_paths),), float(arr), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or rank-1. Got shape={arr.shape}.")
    if arr.shape[0] != int(n_paths):
        raise ValueError(f"{name} length mismatch: expected {int(n_paths)}, got {arr.shape[0]}.")
    return arr.astype(np.float64)


def _student_t_df_from_moments(
    sum2: np.ndarray,
    sum4: np.ndarray,
    count: np.ndarray,
    min_obs: int,
    default_df: float,
    df_floor: float,
    df_cap: float | None,
) -> np.ndarray:
    """
    Method-of-moments Student-t df estimate from standardized shocks.
    Uses excess kurtosis relation: k_excess = 6/(nu-4), nu>4.
    """
    c = np.asarray(count, dtype=np.float64)
    s2 = np.asarray(sum2, dtype=np.float64)
    s4 = np.asarray(sum4, dtype=np.float64)
    out = np.full(c.shape, float(default_df), dtype=np.float64)

    valid = c >= int(min_obs)
    if np.any(valid):
        mean2 = np.divide(s2[valid], c[valid], out=np.zeros_like(s2[valid]), where=c[valid] > 0)
        mean4 = np.divide(s4[valid], c[valid], out=np.zeros_like(s4[valid]), where=c[valid] > 0)
        denom = np.maximum(np.square(mean2), 1e-12)
        excess = np.maximum(mean4 / denom - 3.0, 0.0)
        nu_hat = np.full(excess.shape, float(df_cap if df_cap is not None else 200.0), dtype=np.float64)
        mask_pos = excess > 1e-8
        # For heavy tails (positive excess), infer finite nu.
        nu_hat[mask_pos] = 4.0 + 6.0 / excess[mask_pos]
        out[valid] = nu_hat

    out = _clip_df(out, df_floor=df_floor, df_cap=df_cap)
    return out


def estimate_pathwise_garch_sigma_from_context(
    hedge_paths_2d: np.ndarray,
    pre_history_prices_2d: np.ndarray | None = None,
    context_days: int = 50,
    mode: str = "static",
    trading_days_per_year: int = 252,
    garch_alpha: float = 0.05,
    garch_beta: float = 0.9,
    garch_leverage: float = 0.0,
    garch_omega: float | None = None,
    min_obs: int = 10,
    default_sigma: float = 0.2,
    sigma_floor: float = 1e-6,
    sigma_cap: float | None = None,
):
    """
    Estimate per-path Black-Scholes sigma using a simple GARCH filter over observed history.

    Returns:
    - static mode: np.ndarray shape (n_paths,)
    - stepwise mode: np.ndarray shape (n_paths, n_hedge_steps)
    """
    mode = str(mode).strip().lower()
    if mode not in {"static", "stepwise"}:
        raise ValueError("mode must be 'static' or 'stepwise'.")
    context_days = int(context_days)
    if context_days < 0:
        raise ValueError("context_days must be >= 0.")
    trading_days_per_year = int(trading_days_per_year)
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be > 0.")
    min_obs = int(min_obs)
    if min_obs < 1:
        raise ValueError("min_obs must be >= 1.")

    hedge = np.asarray(hedge_paths_2d, dtype=np.float64)
    if hedge.ndim != 2:
        raise ValueError(f"hedge_paths_2d must be rank-2. Got shape={hedge.shape}.")
    n_paths, n_points = hedge.shape
    if n_points < 2:
        raise ValueError("hedge_paths_2d must have at least 2 points per path.")
    n_steps = int(n_points - 1)

    alpha = _as_path_vector(garch_alpha, n_paths=n_paths, name="garch_alpha")
    beta = _as_path_vector(garch_beta, n_paths=n_paths, name="garch_beta")
    leverage = _as_path_vector(garch_leverage, n_paths=n_paths, name="garch_leverage")
    if np.any(alpha < 0.0) or np.any(beta < 0.0):
        raise ValueError("garch_alpha and garch_beta must be >= 0.")
    if np.any(alpha + beta >= 1.0):
        raise ValueError("Require garch_alpha + garch_beta < 1 for stability.")
    if np.any(alpha + beta + 2.0 * leverage >= 1.0):
        raise ValueError(
            "Require garch_alpha + garch_beta + 2*garch_leverage < 1 for stationarity."
        )

    if pre_history_prices_2d is None:
        pre = np.zeros((n_paths, 0), dtype=np.float64)
    else:
        pre = np.asarray(pre_history_prices_2d, dtype=np.float64)
        if pre.ndim != 2:
            raise ValueError(
                f"pre_history_prices_2d must be rank-2. Got shape={pre.shape}."
            )
        if pre.shape[0] != n_paths:
            raise ValueError(
                f"pre_history_prices_2d n_paths mismatch: expected {n_paths}, got {pre.shape[0]}."
            )

    if context_days > 0 and pre.shape[1] > 0:
        pre_tail = pre[:, -context_days:]
    else:
        pre_tail = np.zeros((n_paths, 0), dtype=np.float64)

    # Context returns are computed from [prehistory, S0], so decision at t=0 only uses prior info.
    if pre_tail.shape[1] > 0:
        context_series = np.concatenate([pre_tail, hedge[:, :1]], axis=1)
    else:
        context_series = hedge[:, :1]

    safe_context = np.maximum(context_series, 1e-12)
    context_lr = np.diff(np.log(safe_context), axis=1)
    scale = float(np.sqrt(trading_days_per_year))
    context_eps = context_lr * scale

    default_var = float(default_sigma) ** 2
    if context_eps.shape[1] > 0:
        obs_counts = np.full((n_paths,), int(context_eps.shape[1]), dtype=np.int32)
        # ddof=1 when possible; ddof=0 for one-observation edge case.
        if context_eps.shape[1] >= 2:
            context_var = np.var(context_eps, axis=1, ddof=1)
        else:
            context_var = np.var(context_eps, axis=1, ddof=0)
    else:
        obs_counts = np.zeros((n_paths,), dtype=np.int32)
        context_var = np.full((n_paths,), np.nan, dtype=np.float64)

    sample_var = np.where(
        obs_counts >= min_obs,
        np.maximum(context_var, 1e-12),
        np.full((n_paths,), max(default_var, 1e-12), dtype=np.float64),
    )

    if garch_omega is None:
        omega_vec = np.maximum((1.0 - alpha - beta) * sample_var, 1e-14)
    else:
        omega_vec = _as_path_vector(garch_omega, n_paths=n_paths, name="garch_omega")
        omega_vec = np.maximum(omega_vec, 1e-14)

    v = sample_var.copy()
    for j in range(context_eps.shape[1]):
        e = context_eps[:, j]
        e2 = np.square(e)
        neg = (e < 0.0).astype(np.float64)
        v = omega_vec + alpha * e2 + leverage * neg * e2 + beta * v
        v = np.maximum(v, 1e-12)

    sigma_t0 = np.sqrt(v)
    sigma_t0 = _clip_sigma(sigma_t0, sigma_floor=sigma_floor, sigma_cap=sigma_cap)

    if mode == "static":
        return sigma_t0.astype(np.float32)

    out = np.zeros((n_paths, n_steps), dtype=np.float64)
    out[:, 0] = sigma_t0

    # For decision at t, include returns observed up to S_t; update using hedge return from t-1 -> t.
    safe_hedge = np.maximum(hedge, 1e-12)
    hedge_lr = np.diff(np.log(safe_hedge), axis=1) * scale  # shape (n_paths, n_steps)
    for t in range(1, n_steps):
        e = hedge_lr[:, t - 1]
        e2 = np.square(e)
        neg = (e < 0.0).astype(np.float64)
        v = omega_vec + alpha * e2 + leverage * neg * e2 + beta * v
        v = np.maximum(v, 1e-12)
        out[:, t] = np.sqrt(v)

    out = _clip_sigma(out, sigma_floor=sigma_floor, sigma_cap=sigma_cap)
    return out.astype(np.float32)


def fit_pathwise_garch_params_from_context(
    hedge_paths_2d: np.ndarray,
    pre_history_prices_2d: np.ndarray | None = None,
    context_days: int = 50,
    trading_days_per_year: int = 252,
    min_obs: int = 10,
    default_alpha: float = 0.05,
    default_beta: float = 0.9,
    default_leverage: float = 0.0,
    leverage_cap: float = 0.25,
    persistence_floor: float = 0.05,
    persistence_cap: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate pathwise GARCH parameters (alpha, beta, leverage) from context returns.

    This is a robust moment-based estimator designed for large path batches:
    - Computes lag-1 autocorrelation of squared returns as persistence proxy.
    - Computes downside-vs-upside squared-return asymmetry as leverage proxy.
    - Maps these moments to stable (alpha, beta, leverage) per path.
    """
    context_days = int(context_days)
    if context_days < 0:
        raise ValueError("context_days must be >= 0.")
    trading_days_per_year = int(trading_days_per_year)
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be > 0.")
    min_obs = int(min_obs)
    if min_obs < 1:
        raise ValueError("min_obs must be >= 1.")

    hedge = np.asarray(hedge_paths_2d, dtype=np.float64)
    if hedge.ndim != 2:
        raise ValueError(f"hedge_paths_2d must be rank-2. Got shape={hedge.shape}.")
    n_paths, n_points = hedge.shape
    if n_points < 2:
        raise ValueError("hedge_paths_2d must have at least 2 points per path.")

    if pre_history_prices_2d is None:
        pre = np.zeros((n_paths, 0), dtype=np.float64)
    else:
        pre = np.asarray(pre_history_prices_2d, dtype=np.float64)
        if pre.ndim != 2:
            raise ValueError(
                f"pre_history_prices_2d must be rank-2. Got shape={pre.shape}."
            )
        if pre.shape[0] != n_paths:
            raise ValueError(
                f"pre_history_prices_2d n_paths mismatch: expected {n_paths}, got {pre.shape[0]}."
            )

    if context_days > 0 and pre.shape[1] > 0:
        pre_tail = pre[:, -context_days:]
    else:
        pre_tail = np.zeros((n_paths, 0), dtype=np.float64)

    if pre_tail.shape[1] > 0:
        context_series = np.concatenate([pre_tail, hedge[:, :1]], axis=1)
    else:
        context_series = hedge[:, :1]

    safe_context = np.maximum(context_series, 1e-12)
    context_lr = np.diff(np.log(safe_context), axis=1)
    scale = float(np.sqrt(trading_days_per_year))
    eps = context_lr * scale
    m = int(eps.shape[1])

    alpha_out = np.full((n_paths,), float(default_alpha), dtype=np.float64)
    beta_out = np.full((n_paths,), float(default_beta), dtype=np.float64)
    leverage_out = np.full((n_paths,), float(default_leverage), dtype=np.float64)

    if m < max(2, min_obs):
        # Not enough context; return defaults (caller can still clip/stability-check).
        return (
            alpha_out.astype(np.float32),
            beta_out.astype(np.float32),
            leverage_out.astype(np.float32),
        )

    e2 = np.square(eps)

    # Persistence proxy via lag-1 corr of squared returns.
    x = e2[:, :-1]
    y = e2[:, 1:]
    x_mu = np.mean(x, axis=1, keepdims=True)
    y_mu = np.mean(y, axis=1, keepdims=True)
    xc = x - x_mu
    yc = y - y_mu
    num = np.sum(xc * yc, axis=1)
    den = np.sqrt(np.sum(np.square(xc), axis=1) * np.sum(np.square(yc), axis=1))
    ac1 = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    ac1 = np.clip(ac1, 0.0, 0.995)

    # Leverage proxy from downside-vs-upside squared shocks.
    neg = eps < 0.0
    pos = ~neg
    neg_count = np.maximum(np.sum(neg, axis=1), 1)
    pos_count = np.maximum(np.sum(pos, axis=1), 1)
    mean_neg = np.sum(e2 * neg, axis=1) / neg_count
    mean_pos = np.sum(e2 * pos, axis=1) / pos_count
    asym = (mean_neg - mean_pos) / np.maximum(mean_neg + mean_pos, 1e-12)
    lev = np.clip(0.20 * asym, 0.0, float(leverage_cap))

    persistence = np.clip(0.15 + 0.80 * ac1, float(persistence_floor), float(persistence_cap))
    beta = 0.80 * persistence
    alpha = persistence - beta

    # Enforce stationarity pathwise.
    stability = alpha + beta + 2.0 * lev
    max_sum = 0.995
    scale_down = np.minimum(1.0, max_sum / np.maximum(stability, 1e-12))
    alpha = alpha * scale_down
    beta = beta * scale_down
    lev = lev * scale_down

    alpha = np.maximum(alpha, 1e-6)
    beta = np.maximum(beta, 1e-6)
    lev = np.maximum(lev, 0.0)

    alpha_out[:] = alpha
    beta_out[:] = beta
    leverage_out[:] = lev
    return (
        alpha_out.astype(np.float32),
        beta_out.astype(np.float32),
        leverage_out.astype(np.float32),
    )


def fit_student_t_df_from_garch_context(
    hedge_paths_2d: np.ndarray,
    pre_history_prices_2d: np.ndarray | None = None,
    context_days: int = 50,
    mode: str = "static",
    trading_days_per_year: int = 252,
    garch_alpha: float = 0.05,
    garch_beta: float = 0.9,
    garch_leverage: float = 0.0,
    garch_omega: float | None = None,
    min_obs: int = 10,
    default_sigma: float = 0.2,
    default_df: float = 8.0,
    df_floor: float = 2.1,
    df_cap: float | None = 200.0,
) -> np.ndarray:
    """
    Fit per-path Student-t degrees of freedom from standardized shocks z_t = eps_t / sqrt(v_t)
    where v_t is obtained by a GARCH filter over context (and optionally hedge history stepwise).

    Returns:
    - static mode: np.ndarray shape (n_paths,)
    - stepwise mode: np.ndarray shape (n_paths, n_hedge_steps)
    """
    mode = str(mode).strip().lower()
    if mode not in {"static", "stepwise"}:
        raise ValueError("mode must be 'static' or 'stepwise'.")
    context_days = int(context_days)
    if context_days < 0:
        raise ValueError("context_days must be >= 0.")
    trading_days_per_year = int(trading_days_per_year)
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be > 0.")
    min_obs = int(min_obs)
    if min_obs < 1:
        raise ValueError("min_obs must be >= 1.")
    default_df = float(default_df)
    if default_df <= 2.0:
        raise ValueError("default_df must be > 2.")
    df_floor = float(df_floor)
    if df_floor <= 2.0:
        raise ValueError("df_floor must be > 2.")
    if df_cap is not None and float(df_cap) <= df_floor:
        raise ValueError("df_cap must be > df_floor when provided.")

    hedge = np.asarray(hedge_paths_2d, dtype=np.float64)
    if hedge.ndim != 2:
        raise ValueError(f"hedge_paths_2d must be rank-2. Got shape={hedge.shape}.")
    n_paths, n_points = hedge.shape
    if n_points < 2:
        raise ValueError("hedge_paths_2d must have at least 2 points per path.")
    n_steps = int(n_points - 1)

    alpha = _as_path_vector(garch_alpha, n_paths=n_paths, name="garch_alpha")
    beta = _as_path_vector(garch_beta, n_paths=n_paths, name="garch_beta")
    leverage = _as_path_vector(garch_leverage, n_paths=n_paths, name="garch_leverage")
    if np.any(alpha < 0.0) or np.any(beta < 0.0):
        raise ValueError("garch_alpha and garch_beta must be >= 0.")
    if np.any(alpha + beta >= 1.0):
        raise ValueError("Require garch_alpha + garch_beta < 1 for stability.")
    if np.any(alpha + beta + 2.0 * leverage >= 1.0):
        raise ValueError(
            "Require garch_alpha + garch_beta + 2*garch_leverage < 1 for stationarity."
        )

    if pre_history_prices_2d is None:
        pre = np.zeros((n_paths, 0), dtype=np.float64)
    else:
        pre = np.asarray(pre_history_prices_2d, dtype=np.float64)
        if pre.ndim != 2:
            raise ValueError(
                f"pre_history_prices_2d must be rank-2. Got shape={pre.shape}."
            )
        if pre.shape[0] != n_paths:
            raise ValueError(
                f"pre_history_prices_2d n_paths mismatch: expected {n_paths}, got {pre.shape[0]}."
            )

    if context_days > 0 and pre.shape[1] > 0:
        pre_tail = pre[:, -context_days:]
    else:
        pre_tail = np.zeros((n_paths, 0), dtype=np.float64)

    if pre_tail.shape[1] > 0:
        context_series = np.concatenate([pre_tail, hedge[:, :1]], axis=1)
    else:
        context_series = hedge[:, :1]

    safe_context = np.maximum(context_series, 1e-12)
    context_lr = np.diff(np.log(safe_context), axis=1)
    scale = float(np.sqrt(trading_days_per_year))
    context_eps = context_lr * scale

    default_var = float(default_sigma) ** 2
    if context_eps.shape[1] > 0:
        obs_counts = np.full((n_paths,), int(context_eps.shape[1]), dtype=np.int32)
        if context_eps.shape[1] >= 2:
            context_var = np.var(context_eps, axis=1, ddof=1)
        else:
            context_var = np.var(context_eps, axis=1, ddof=0)
    else:
        obs_counts = np.zeros((n_paths,), dtype=np.int32)
        context_var = np.full((n_paths,), np.nan, dtype=np.float64)

    sample_var = np.where(
        obs_counts >= min_obs,
        np.maximum(context_var, 1e-12),
        np.full((n_paths,), max(default_var, 1e-12), dtype=np.float64),
    )

    if garch_omega is None:
        omega_vec = np.maximum((1.0 - alpha - beta) * sample_var, 1e-14)
    else:
        omega_vec = _as_path_vector(garch_omega, n_paths=n_paths, name="garch_omega")
        omega_vec = np.maximum(omega_vec, 1e-14)

    v = sample_var.copy()
    z_sum2 = np.zeros((n_paths,), dtype=np.float64)
    z_sum4 = np.zeros((n_paths,), dtype=np.float64)
    z_count = np.zeros((n_paths,), dtype=np.int32)
    for j in range(context_eps.shape[1]):
        e = context_eps[:, j]
        z = e / np.sqrt(np.maximum(v, 1e-12))
        z2 = np.square(z)
        z_sum2 += z2
        z_sum4 += np.square(z2)
        z_count += 1

        e2 = np.square(e)
        neg = (e < 0.0).astype(np.float64)
        v = omega_vec + alpha * e2 + leverage * neg * e2 + beta * v
        v = np.maximum(v, 1e-12)

    if mode == "static":
        df_hat = _student_t_df_from_moments(
            sum2=z_sum2,
            sum4=z_sum4,
            count=z_count,
            min_obs=min_obs,
            default_df=default_df,
            df_floor=df_floor,
            df_cap=df_cap,
        )
        return df_hat.astype(np.float32)

    out = np.zeros((n_paths, n_steps), dtype=np.float64)
    out[:, 0] = _student_t_df_from_moments(
        sum2=z_sum2,
        sum4=z_sum4,
        count=z_count,
        min_obs=min_obs,
        default_df=default_df,
        df_floor=df_floor,
        df_cap=df_cap,
    )

    safe_hedge = np.maximum(hedge, 1e-12)
    hedge_lr = np.diff(np.log(safe_hedge), axis=1) * scale
    for t in range(1, n_steps):
        e = hedge_lr[:, t - 1]
        z = e / np.sqrt(np.maximum(v, 1e-12))
        z2 = np.square(z)
        z_sum2 += z2
        z_sum4 += np.square(z2)
        z_count += 1

        e2 = np.square(e)
        neg = (e < 0.0).astype(np.float64)
        v = omega_vec + alpha * e2 + leverage * neg * e2 + beta * v
        v = np.maximum(v, 1e-12)
        out[:, t] = _student_t_df_from_moments(
            sum2=z_sum2,
            sum4=z_sum4,
            count=z_count,
            min_obs=min_obs,
            default_df=default_df,
            df_floor=df_floor,
            df_cap=df_cap,
        )

    out = _clip_df(out, df_floor=df_floor, df_cap=df_cap)
    return out.astype(np.float32)
