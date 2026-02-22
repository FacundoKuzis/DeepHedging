import tensorflow as tf


def _validate_paths(paths: tf.Tensor, underlying_index: int) -> tf.Tensor:
    x = tf.convert_to_tensor(paths, dtype=tf.float32)
    if len(x.shape) != 3:
        raise ValueError(f"paths must be rank-3 (batch, timesteps, instruments). Got shape={x.shape}.")
    n_instr = int(x.shape[2])
    if underlying_index < 0 or underlying_index >= n_instr:
        raise ValueError(
            f"underlying_index out of bounds: got {underlying_index}, n_instruments={n_instr}."
        )
    if int(x.shape[1]) < 2:
        raise ValueError("paths must contain at least 2 timesteps.")
    return x


def _build_causal_lag_matrix(seq_2d: tf.Tensor, context_length: int) -> tf.Tensor:
    """
    Build causal lag features from a sequence tensor.

    Args:
        seq_2d: Tensor shape (batch, n_steps).
        context_length: Number of lag values per step.

    Returns:
        Tensor shape (batch, n_steps, context_length), ordered oldest->newest.
    """
    if int(context_length) <= 0:
        batch = tf.shape(seq_2d)[0]
        n_steps = tf.shape(seq_2d)[1]
        return tf.zeros((batch, n_steps, 0), dtype=tf.float32)

    seq = tf.convert_to_tensor(seq_2d, dtype=tf.float32)
    batch = tf.shape(seq)[0]
    n_steps = tf.shape(seq)[1]
    lags = []
    # oldest -> newest
    for lag in range(int(context_length) - 1, -1, -1):
        if lag == 0:
            shifted = seq
        else:
            prefix = seq[:, :-lag]
            shifted = tf.pad(prefix, paddings=[[0, 0], [lag, 0]], constant_values=0.0)
            # Keep a fixed temporal length for every lag even when lag > n_steps.
            shifted = shifted[:, :n_steps]
        lags.append(shifted)
    return tf.stack(lags, axis=-1)


def build_causal_history_features(
    paths: tf.Tensor,
    context_length: int,
    feature_mode: str,
    strike: float | None = None,
    underlying_index: int = 0,
    pre_history_prices: tf.Tensor | None = None,
) -> tf.Tensor | None:
    """
    Build causal history features for hedge decisions at timesteps 0..N-1.

    Given paths shape (batch, N+1, n_instruments), returns features shape
    (batch, N, context_length). Feature options:
    - log_returns: causal lagged log-returns known at each decision time.
    - log_moneyness: causal lagged log(S_t / K) known at each decision time.
    """
    context_length = int(context_length)
    if context_length <= 0:
        return None

    mode = str(feature_mode).strip().lower()
    x = _validate_paths(paths, underlying_index=underlying_index)
    eps = tf.constant(1e-8, dtype=tf.float32)
    prices = tf.maximum(x[:, :, underlying_index], eps)  # (batch, N+1)

    if mode == "log_returns":
        # Hedge-window returns: r1..rN where rj = log(S_j / S_{j-1}).
        rets = tf.math.log(prices[:, 1:] / prices[:, :-1])  # (batch, N)

        if pre_history_prices is not None:
            pre = tf.convert_to_tensor(pre_history_prices, dtype=tf.float32)
            if len(pre.shape) != 2:
                raise ValueError(
                    f"pre_history_prices must be rank-2 (batch, history_steps). Got {pre.shape}."
                )
            if int(pre.shape[0]) != int(prices.shape[0]):
                raise ValueError(
                    "pre_history_prices batch size mismatch: "
                    f"expected {int(prices.shape[0])}, got {int(pre.shape[0])}."
                )
            pre = tf.maximum(pre, eps)
            # Build pre-window returns including r0 = log(S0/S_-1).
            full_pre_prices = tf.concat([pre, prices[:, :1]], axis=1)  # (..., S_-1, S0)
            pre_rets = tf.math.log(full_pre_prices[:, 1:] / full_pre_prices[:, :-1])  # (batch, L)
            # Current known return at decision t: [r0, r1, ..., r_{N-1}]
            current = tf.concat([pre_rets[:, -1:], rets[:, :-1]], axis=1)  # (batch, N)
            older = pre_rets[:, :-1]  # (batch, L-1), may be empty
            if int(older.shape[1]) > 0:
                combined = tf.concat([older, current], axis=1)
                all_lags = _build_causal_lag_matrix(combined, context_length=context_length)
                return all_lags[:, -int(current.shape[1]) :, :]
            return _build_causal_lag_matrix(current, context_length=context_length)

        # No pre-history available: fallback keeps strict causality with zero bootstrapping.
        known = tf.concat([tf.zeros_like(rets[:, :1]), rets[:, :-1]], axis=1)  # (batch, N)
        return _build_causal_lag_matrix(known, context_length=context_length)

    if mode == "log_moneyness":
        if strike is None:
            raise ValueError("strike is required when feature_mode='log_moneyness'.")
        k = tf.constant(max(float(strike), 1e-8), dtype=tf.float32)
        # At decision t we know S_t. Use prices[:, :-1] -> t=0..N-1.
        current = tf.math.log(prices[:, :-1] / k)  # (batch, N)
        if pre_history_prices is not None:
            pre = tf.convert_to_tensor(pre_history_prices, dtype=tf.float32)
            if len(pre.shape) != 2:
                raise ValueError(
                    f"pre_history_prices must be rank-2 (batch, history_steps). Got {pre.shape}."
                )
            if int(pre.shape[0]) != int(prices.shape[0]):
                raise ValueError(
                    "pre_history_prices batch size mismatch: "
                    f"expected {int(prices.shape[0])}, got {int(pre.shape[0])}."
                )
            pre = tf.maximum(pre, eps)
            older = tf.math.log(pre / k)  # (batch, L)
            combined = tf.concat([older, current], axis=1)
            all_lags = _build_causal_lag_matrix(combined, context_length=context_length)
            return all_lags[:, -int(current.shape[1]) :, :]
        return _build_causal_lag_matrix(current, context_length=context_length)

    raise ValueError(
        f"Unsupported feature_mode='{feature_mode}'. Use 'log_returns' or 'log_moneyness'."
    )
