import numpy as np


def _clip_sigma(sigma, sigma_floor, sigma_cap):
    out = np.maximum(np.asarray(sigma, dtype=np.float64), float(sigma_floor))
    if sigma_cap is not None:
        out = np.minimum(out, float(sigma_cap))
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

    alpha = float(garch_alpha)
    beta = float(garch_beta)
    leverage = float(garch_leverage)
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("garch_alpha and garch_beta must be >= 0.")

    hedge = np.asarray(hedge_paths_2d, dtype=np.float64)
    if hedge.ndim != 2:
        raise ValueError(f"hedge_paths_2d must be rank-2. Got shape={hedge.shape}.")
    n_paths, n_points = hedge.shape
    if n_points < 2:
        raise ValueError("hedge_paths_2d must have at least 2 points per path.")
    n_steps = int(n_points - 1)

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
        omega_vec = np.full((n_paths,), float(garch_omega), dtype=np.float64)
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
