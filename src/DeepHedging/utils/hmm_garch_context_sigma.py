import numpy as np

from DeepHedging.utils.garch_context_sigma import (
    _as_path_vector,
    _clip_sigma,
    estimate_pathwise_garch_sigma_from_context,
    fit_pathwise_garch_params_from_context,
    fit_student_t_df_from_garch_context,
)


def estimate_pathwise_hmm_garch_student_sigma_from_context(
    hedge_paths_2d: np.ndarray,
    pre_history_prices_2d: np.ndarray | None = None,
    context_days: int = 50,
    trading_days_per_year: int = 252,
    garch_alpha: float = 0.05,
    garch_beta: float = 0.9,
    garch_leverage: float = 0.0,
    garch_omega: float | None = None,
    fit_garch_params_from_context: bool = True,
    min_obs: int = 10,
    default_sigma: float = 0.2,
    sigma_floor: float = 1e-6,
    sigma_cap: float | None = None,
    student_t_min_obs: int = 10,
    student_t_df_default: float = 8.0,
    student_t_df_floor: float = 2.1,
    student_t_df_cap: float | None = 200.0,
    hmm_num_states: int = 3,
    hmm_transition_smoothing: float = 1.0,
    hmm_state_multiplier_floor: float = 0.35,
    hmm_state_multiplier_cap: float = 3.5,
    hmm_tail_adjustment_enabled: bool = True,
    hmm_tail_multiplier_cap: float = 1.6,
) -> dict[str, np.ndarray]:
    """
    Stepwise benchmark sigma estimator that combines:
    1) pathwise GARCH filtering
    2) pathwise Student-t df fitting
    3) pathwise regime/state estimation over sigma dynamics (HMM-style).

    Returns dict with:
    - sigma_stepwise: combined sigma forecast used by benchmarks, shape (n_paths, n_steps)
    - sigma_garch_stepwise: pure GARCH sigma forecast, shape (n_paths, n_steps)
    - student_t_df_stepwise: fitted df per path/step, shape (n_paths, n_steps)
    - hmm_states_stepwise: inferred state index per path/step, shape (n_paths, n_steps)
    - hmm_expected_multiplier_stepwise: multiplier applied to sigma_garch, shape (n_paths, n_steps)
    - hmm_transition_probs_last: next-state probs for final step, shape (n_paths, n_states)
    """
    k = int(hmm_num_states)
    if k < 2:
        raise ValueError("hmm_num_states must be >= 2.")
    smoothing = float(hmm_transition_smoothing)
    if smoothing < 0.0:
        raise ValueError("hmm_transition_smoothing must be >= 0.")
    mult_floor = float(hmm_state_multiplier_floor)
    mult_cap = float(hmm_state_multiplier_cap)
    if mult_floor <= 0.0 or mult_cap <= mult_floor:
        raise ValueError(
            "Require 0 < hmm_state_multiplier_floor < hmm_state_multiplier_cap."
        )
    tail_cap = float(hmm_tail_multiplier_cap)
    if tail_cap < 1.0:
        raise ValueError("hmm_tail_multiplier_cap must be >= 1.")

    hedge = np.asarray(hedge_paths_2d, dtype=np.float64)
    if hedge.ndim != 2:
        raise ValueError(f"hedge_paths_2d must be rank-2. Got shape={hedge.shape}.")
    n_paths, n_points = hedge.shape
    if n_points < 2:
        raise ValueError("hedge_paths_2d must have at least 2 points per path.")
    n_steps = int(n_points - 1)

    pre_hist = None if pre_history_prices_2d is None else np.asarray(pre_history_prices_2d, dtype=np.float64)
    if pre_hist is not None:
        if pre_hist.ndim != 2:
            raise ValueError(
                f"pre_history_prices_2d must be rank-2. Got shape={pre_hist.shape}."
            )
        if pre_hist.shape[0] != n_paths:
            raise ValueError(
                f"pre_history_prices_2d n_paths mismatch: expected {n_paths}, got {pre_hist.shape[0]}."
            )

    # Accept scalar or per-path arrays for GARCH params.
    alpha_input_vec = _as_path_vector(garch_alpha, n_paths=n_paths, name="garch_alpha").astype(np.float32)
    beta_input_vec = _as_path_vector(garch_beta, n_paths=n_paths, name="garch_beta").astype(np.float32)
    leverage_input_vec = _as_path_vector(garch_leverage, n_paths=n_paths, name="garch_leverage").astype(np.float32)

    if bool(fit_garch_params_from_context):
        alpha_vec, beta_vec, leverage_vec = fit_pathwise_garch_params_from_context(
            hedge_paths_2d=hedge,
            pre_history_prices_2d=pre_hist,
            context_days=int(context_days),
            trading_days_per_year=int(trading_days_per_year),
            min_obs=int(min_obs),
            # When caller passes per-path defaults, use their mean as robust scalar fallback.
            default_alpha=float(np.mean(alpha_input_vec)),
            default_beta=float(np.mean(beta_input_vec)),
            default_leverage=float(np.mean(leverage_input_vec)),
        )
    else:
        alpha_vec = alpha_input_vec
        beta_vec = beta_input_vec
        leverage_vec = leverage_input_vec

    sigma_garch = estimate_pathwise_garch_sigma_from_context(
        hedge_paths_2d=hedge,
        pre_history_prices_2d=pre_hist,
        context_days=int(context_days),
        mode="stepwise",
        trading_days_per_year=int(trading_days_per_year),
        garch_alpha=alpha_vec,
        garch_beta=beta_vec,
        garch_leverage=leverage_vec,
        garch_omega=None if garch_omega is None else float(garch_omega),
        min_obs=int(min_obs),
        default_sigma=float(default_sigma),
        sigma_floor=float(sigma_floor),
        sigma_cap=None if sigma_cap is None else float(sigma_cap),
    ).astype(np.float64)

    df_step = fit_student_t_df_from_garch_context(
        hedge_paths_2d=hedge,
        pre_history_prices_2d=pre_hist,
        context_days=int(context_days),
        mode="stepwise",
        trading_days_per_year=int(trading_days_per_year),
        garch_alpha=alpha_vec,
        garch_beta=beta_vec,
        garch_leverage=leverage_vec,
        garch_omega=None if garch_omega is None else float(garch_omega),
        min_obs=int(student_t_min_obs),
        default_sigma=float(default_sigma),
        default_df=float(student_t_df_default),
        df_floor=float(student_t_df_floor),
        df_cap=None if student_t_df_cap is None else float(student_t_df_cap),
    ).astype(np.float64)

    states = np.zeros((n_paths, n_steps), dtype=np.int32)
    trans_counts = np.zeros((n_paths, k, k), dtype=np.float64)
    state_sigma_sum = np.zeros((n_paths, k), dtype=np.float64)
    state_sigma_count = np.zeros((n_paths, k), dtype=np.float64)

    sigma_combined = np.zeros_like(sigma_garch, dtype=np.float64)
    expected_multiplier = np.ones_like(sigma_garch, dtype=np.float64)

    quantile_probs = np.linspace(1.0 / float(k), (float(k) - 1.0) / float(k), k - 1)

    for t in range(n_steps):
        current_sigma = sigma_garch[:, t]
        if t == 0:
            state_t = np.full((n_paths,), int(k // 2), dtype=np.int32)
        else:
            hist_sigma = sigma_garch[:, : t + 1]
            thresholds = np.quantile(hist_sigma, quantile_probs, axis=1).T
            state_t = np.sum(current_sigma[:, None] > thresholds, axis=1).astype(np.int32)
            state_t = np.clip(state_t, 0, k - 1)
            prev_state = states[:, t - 1]
            trans_counts[np.arange(n_paths), prev_state, state_t] += 1.0

        states[:, t] = state_t
        state_sigma_sum[np.arange(n_paths), state_t] += current_sigma
        state_sigma_count[np.arange(n_paths), state_t] += 1.0

        overall_mean = np.mean(sigma_garch[:, : t + 1], axis=1, keepdims=True)
        overall_mean = np.maximum(overall_mean, 1e-12)
        state_mean = np.divide(
            state_sigma_sum,
            np.maximum(state_sigma_count, 1e-12),
            out=np.ones_like(state_sigma_sum),
            where=state_sigma_count > 0.0,
        )
        multipliers = np.divide(
            state_mean,
            overall_mean,
            out=np.ones_like(state_mean),
            where=np.isfinite(state_mean),
        )
        multipliers = np.clip(multipliers, mult_floor, mult_cap)

        if t == 0:
            next_probs = np.zeros((n_paths, k), dtype=np.float64)
            next_probs[np.arange(n_paths), state_t] = 1.0
        else:
            row = trans_counts[np.arange(n_paths), state_t, :]
            row = row + float(smoothing)
            row_sum = np.sum(row, axis=1, keepdims=True)
            next_probs = row / np.maximum(row_sum, 1e-12)

        mult_t = np.sum(next_probs * multipliers, axis=1)
        mult_t = np.clip(mult_t, mult_floor, mult_cap)
        expected_multiplier[:, t] = mult_t
        sigma_t = current_sigma * mult_t

        if bool(hmm_tail_adjustment_enabled):
            df_t = np.maximum(df_step[:, t], 2.001)
            tail_mult = np.sqrt(df_t / np.maximum(df_t - 2.0, 1e-6))
            tail_mult = np.clip(tail_mult, 1.0, tail_cap)
            sigma_t = sigma_t * tail_mult

        sigma_combined[:, t] = sigma_t

    sigma_combined = _clip_sigma(
        sigma_combined,
        sigma_floor=float(sigma_floor),
        sigma_cap=None if sigma_cap is None else float(sigma_cap),
    )

    final_state = states[:, -1]
    final_row = trans_counts[np.arange(n_paths), final_state, :] + float(smoothing)
    final_row = final_row / np.maximum(np.sum(final_row, axis=1, keepdims=True), 1e-12)

    return {
        "sigma_stepwise": sigma_combined.astype(np.float32),
        "sigma_garch_stepwise": sigma_garch.astype(np.float32),
        "student_t_df_stepwise": df_step.astype(np.float32),
        "hmm_states_stepwise": states.astype(np.int32),
        "hmm_expected_multiplier_stepwise": expected_multiplier.astype(np.float32),
        "hmm_transition_probs_last": final_row.astype(np.float32),
        "garch_alpha": np.asarray(alpha_vec, dtype=np.float32),
        "garch_beta": np.asarray(beta_vec, dtype=np.float32),
        "garch_leverage": np.asarray(leverage_vec, dtype=np.float32),
    }
