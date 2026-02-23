from __future__ import annotations

from typing import Literal

import numpy as np

from DeepHedging.utils.lrm_continuation import ContinuationValueProvider


def _resolve_vector(values: np.ndarray | float | None, batch: int, default: float) -> np.ndarray:
    if values is None:
        return np.full((batch,), float(default), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((batch,), float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape[0] != batch:
        raise ValueError(f"Vector length mismatch: expected {batch}, got {arr.shape[0]}.")
    return arr


def _outer_normals(
    batch: int,
    n_outer: int,
    rng: np.random.Generator,
    mode: Literal["shared_crn", "per_state"] = "shared_crn",
    use_antithetic: bool = True,
) -> np.ndarray:
    mode = str(mode).strip().lower()  # type: ignore[assignment]
    if mode not in {"shared_crn", "per_state"}:
        raise ValueError("seed_mode must be 'shared_crn' or 'per_state'.")

    half = (int(n_outer) + 1) // 2 if use_antithetic else int(n_outer)
    if mode == "shared_crn":
        base = rng.normal(0.0, 1.0, size=(1, half))
        if use_antithetic:
            z = np.concatenate([base, -base], axis=1)[:, : int(n_outer)]
        else:
            z = base
        return np.repeat(z, batch, axis=0)

    base = rng.normal(0.0, 1.0, size=(batch, half))
    if use_antithetic:
        return np.concatenate([base, -base], axis=1)[:, : int(n_outer)]
    return base


def compute_lrm_target_batch(
    spot_t: np.ndarray,
    t_index: int,
    dt: float,
    provider: ContinuationValueProvider,
    per_path_r: np.ndarray | float | None,
    per_path_sigma: np.ndarray | float | None,
    path_prefix: np.ndarray | None = None,
    outer_paths: int = 512,
    var_epsilon: float = 1e-10,
    use_antithetic: bool = True,
    seed: int | None = None,
    seed_mode: Literal["shared_crn", "per_state"] = "shared_crn",
) -> np.ndarray:
    """
    Compute local risk minimization hedge targets h_t pathwise.

    h_t = Cov(C_{t+1}, dS_{t+1}) / Var(dS_{t+1}),
    with dS_{t+1} = S_{t+1} - S_t * exp(r*dt).
    """
    s_t = np.asarray(spot_t, dtype=np.float64).reshape(-1)
    batch = int(s_t.shape[0])
    if batch == 0:
        return np.empty((0,), dtype=np.float32)

    if int(outer_paths) <= 1:
        raise ValueError("outer_paths must be > 1.")
    if float(dt) <= 0.0:
        raise ValueError("dt must be > 0.")

    if provider.context is None:
        raise ValueError("Continuation provider is not prepared. Call provider.prepare(context) first.")

    r_vec = _resolve_vector(per_path_r, batch=batch, default=float(provider.context.instrument.r))
    sigma_vec = np.maximum(
        _resolve_vector(per_path_sigma, batch=batch, default=float(provider.context.instrument.sigma)),
        1e-8,
    )

    rng = np.random.default_rng(None if seed is None else int(seed))
    z = _outer_normals(
        batch=batch,
        n_outer=int(outer_paths),
        rng=rng,
        mode=seed_mode,
        use_antithetic=bool(use_antithetic),
    )
    drift = (r_vec[:, None] - 0.5 * np.square(sigma_vec[:, None])) * float(dt)
    diffusion = sigma_vec[:, None] * np.sqrt(float(dt)) * z
    s_t1 = s_t[:, None] * np.exp(drift + diffusion)

    continuation = provider.estimate_continuation_t1(
        spot_t1=s_t1,
        t_index=int(t_index),
        path_prefix=path_prefix,
        per_path_r=r_vec,
        per_path_sigma=sigma_vec,
        seed=seed,
    )
    continuation = np.asarray(continuation, dtype=np.float64)
    if continuation.shape != s_t1.shape:
        raise ValueError(
            "Continuation provider returned invalid shape: "
            f"expected {s_t1.shape}, got {continuation.shape}."
        )

    d_s = s_t1 - s_t[:, None] * np.exp(r_vec[:, None] * float(dt))

    d_s_centered = d_s - d_s.mean(axis=1, keepdims=True)
    c_centered = continuation - continuation.mean(axis=1, keepdims=True)
    cov = np.mean(d_s_centered * c_centered, axis=1)
    var = np.mean(np.square(d_s_centered), axis=1)
    hedge = cov / np.maximum(var, float(var_epsilon))
    return hedge.astype(np.float32)
