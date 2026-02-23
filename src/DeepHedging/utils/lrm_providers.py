from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
import time
from typing import Callable

import numpy as np
import tensorflow as tf

from DeepHedging.ContingentClaims import (
    AsianArithmeticCall,
    AsianArithmeticPut,
    AsianGeometricCall,
    AsianGeometricPut,
)
from DeepHedging.ContingentClaims import EuropeanCall, EuropeanPut
from DeepHedging.utils.asian_pricing import _geometric_conditional_price_np_batch
from DeepHedging.utils.lrm_continuation import ContinuationContext, ContinuationValueProvider


ASIAN_CLAIMS = (
    AsianArithmeticCall,
    AsianArithmeticPut,
    AsianGeometricCall,
    AsianGeometricPut,
)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """
    Fast normal CDF approximation (Abramowitz-Stegun style).
    Avoids SciPy dependency and works reliably across numpy builds.
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


def _bs_price_batch(
    option_type: str,
    spot: np.ndarray,
    strike: float,
    rate: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    s = np.maximum(np.asarray(spot, dtype=np.float64), 1e-12)
    k = float(strike)
    r = np.asarray(rate, dtype=np.float64)
    v = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    t = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)

    out = np.empty_like(s)
    immediate = t <= 1e-12
    if np.any(immediate):
        if option_type == "call":
            out[immediate] = np.maximum(s[immediate] - k, 0.0)
        else:
            out[immediate] = np.maximum(k - s[immediate], 0.0)

    alive = ~immediate
    if np.any(alive):
        sa = s[alive]
        ra = r[alive]
        va = v[alive]
        ta = t[alive]
        sqrt_t = np.sqrt(ta)
        d1 = (np.log(sa / k) + (ra + 0.5 * va * va) * ta) / (va * sqrt_t)
        d2 = d1 - va * sqrt_t
        if option_type == "call":
            out[alive] = sa * _normal_cdf(d1) - k * np.exp(-ra * ta) * _normal_cdf(d2)
        else:
            out[alive] = k * np.exp(-ra * ta) * _normal_cdf(-d2) - sa * _normal_cdf(-d1)
    return np.maximum(out, 0.0)


def _simulate_gbm_segment(
    s_start: np.ndarray,
    r_vec: np.ndarray,
    sigma_vec: np.ndarray,
    dt: float,
    n_steps: int,
    rng: np.random.Generator,
    n_scenarios: int,
    use_antithetic: bool,
) -> np.ndarray:
    """
    Returns paths segment including start point.
    Shape: (n_states, n_scenarios, n_steps+1)
    """
    n_states = int(s_start.shape[0])
    if n_steps == 0:
        return np.repeat(s_start[:, None, None], n_scenarios, axis=1)

    half = (int(n_scenarios) + 1) // 2 if use_antithetic else int(n_scenarios)
    z = rng.normal(0.0, 1.0, size=(n_states, half, int(n_steps)))
    if use_antithetic:
        z = np.concatenate([z, -z], axis=1)[:, : int(n_scenarios), :]

    drift = (r_vec[:, None, None] - 0.5 * np.square(sigma_vec[:, None, None])) * float(dt)
    diff = sigma_vec[:, None, None] * np.sqrt(float(dt)) * z
    log_inc = drift + diff
    log_cum = np.cumsum(log_inc, axis=2)
    future = s_start[:, None, None] * np.exp(log_cum)
    start = np.repeat(s_start[:, None, None], int(n_scenarios), axis=1)
    return np.concatenate([start, future], axis=2)


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = X.T @ X
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float64)
    xty = X.T @ y
    return np.linalg.solve(xtx + reg, xty)


@dataclass
class _LsmModel:
    betas_by_step: dict[int, np.ndarray]
    feature_dim: int


class BSClosedFormContinuationProvider(ContinuationValueProvider):
    provider_name = "bs_closed_form"

    def supports_claim(self, claim) -> bool:
        return isinstance(claim, (EuropeanCall, EuropeanPut))

    def estimate_continuation_t1(
        self,
        spot_t1: np.ndarray,
        t_index: int,
        path_prefix: np.ndarray | None = None,
        per_path_r: np.ndarray | float | None = None,
        per_path_sigma: np.ndarray | float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        _ = path_prefix
        _ = seed
        if self.context is None:
            raise ValueError("Provider must be prepared before estimation.")

        claim = self.context.claim
        if not self.supports_claim(claim):
            raise ValueError(
                "BSClosedFormContinuationProvider supports only EuropeanCall/EuropeanPut."
            )

        spot = np.asarray(spot_t1, dtype=np.float64)
        if spot.ndim != 2:
            raise ValueError(f"spot_t1 must have shape (batch, n_outer). Got {spot.shape}")
        batch = int(spot.shape[0])

        r_vec = _resolve_vector(per_path_r, batch=batch, default=float(self.context.instrument.r))
        sigma_vec = np.maximum(
            _resolve_vector(per_path_sigma, batch=batch, default=float(self.context.instrument.sigma)),
            1e-8,
        )
        tau = max((int(self.context.n_steps) - (int(t_index) + 1)) * float(self.context.dt), 0.0)

        option_type = str(getattr(claim, "option_type", "call")).strip().lower()
        r_full = np.repeat(r_vec[:, None], spot.shape[1], axis=1)
        sigma_full = np.repeat(sigma_vec[:, None], spot.shape[1], axis=1)
        tau_full = np.full_like(spot, float(tau), dtype=np.float64)
        return _bs_price_batch(
            option_type=option_type,
            spot=spot,
            strike=float(claim.strike),
            rate=r_full,
            sigma=sigma_full,
            tau=tau_full,
        ).astype(np.float32)

    def get_price_batch(
        self,
        path_s0: np.ndarray,
        path_r: np.ndarray | float,
        path_sigma: np.ndarray | float,
    ) -> np.ndarray:
        if self.context is None:
            raise ValueError("Provider must be prepared before pricing.")
        claim = self.context.claim
        if not self.supports_claim(claim):
            raise ValueError(
                "BSClosedFormContinuationProvider supports only EuropeanCall/EuropeanPut."
            )
        s0 = np.asarray(path_s0, dtype=np.float64).reshape(-1)
        batch = int(s0.shape[0])
        r_vec = _resolve_vector(path_r, batch=batch, default=float(self.context.instrument.r))
        sigma_vec = np.maximum(
            _resolve_vector(path_sigma, batch=batch, default=float(self.context.instrument.sigma)),
            1e-8,
        )
        tau = np.full((batch,), float(self.context.maturity), dtype=np.float64)
        option_type = str(getattr(claim, "option_type", "call")).strip().lower()
        return _bs_price_batch(
            option_type=option_type,
            spot=s0,
            strike=float(claim.strike),
            rate=r_vec,
            sigma=sigma_vec,
            tau=tau,
        ).astype(np.float32)


class MonteCarloContinuationProvider(ContinuationValueProvider):
    provider_name = "monte_carlo"

    def __init__(
        self,
        inner_paths: int = 1024,
        inner_chunk_size: int = 64,
        parallel_enabled: bool = False,
        n_workers: int = 1,
        parallel_backend: str = "thread",
        parallel_chunk_size: int | None = None,
        use_antithetic: bool = True,
        verbose: bool = False,
        log_every_chunks: int = 0,
    ) -> None:
        super().__init__()
        self.inner_paths = max(2, int(inner_paths))
        self.inner_chunk_size = max(1, int(inner_chunk_size))
        self.parallel_enabled = bool(parallel_enabled)
        self.n_workers = max(1, int(n_workers))
        self.parallel_backend = str(parallel_backend).strip().lower()
        if self.parallel_backend not in {"thread", "process"}:
            raise ValueError("parallel_backend must be 'thread' or 'process'.")
        self.parallel_chunk_size = (
            None if parallel_chunk_size is None else max(1, int(parallel_chunk_size))
        )
        self.use_antithetic = bool(use_antithetic)
        self.verbose = bool(verbose)
        self.log_every_chunks = max(0, int(log_every_chunks))

    def _mc_log(self, message: str):
        if not self.verbose:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [lrm][mc] {message}")

    def _estimate_states_chunk(
        self,
        state_spot: np.ndarray,
        state_prefix: np.ndarray,
        state_r: np.ndarray,
        state_sigma: np.ndarray,
        t_index: int,
        seed: int,
    ) -> np.ndarray:
        if self.context is None:
            raise ValueError("Provider must be prepared before estimation.")

        n_states = int(state_spot.shape[0])
        n_steps = int(self.context.n_steps)
        dt = float(self.context.dt)
        claim = self.context.claim

        n_remaining = n_steps - (int(t_index) + 1)
        if n_remaining < 0:
            raise ValueError("Invalid t_index for continuation estimation.")

        rng = np.random.default_rng(int(seed))

        segment = _simulate_gbm_segment(
            s_start=state_spot,
            r_vec=state_r,
            sigma_vec=state_sigma,
            dt=dt,
            n_steps=n_remaining,
            rng=rng,
            n_scenarios=self.inner_paths,
            use_antithetic=self.use_antithetic,
        )  # (n_states, inner, n_remaining+1)

        prefix_rep = np.repeat(state_prefix[:, None, :], self.inner_paths, axis=1)
        full_paths = np.concatenate([prefix_rep, segment], axis=2)
        full_flat = full_paths.reshape(-1, n_steps + 1).astype(np.float32)

        payoff = claim.calculate_payoff(tf.convert_to_tensor(full_flat, dtype=tf.float32))
        payoff_np = np.asarray(payoff.numpy(), dtype=np.float64).reshape(n_states, self.inner_paths)

        disc = np.exp(-state_r * float(n_remaining) * dt)
        continuation = disc[:, None] * payoff_np
        return continuation.mean(axis=1)

    def _resolve_state_chunk_size(self, n_states: int) -> int:
        n_states = int(max(1, n_states))
        # Backward-compatible baseline for non-parallel mode.
        if not (self.parallel_enabled and self.n_workers > 1):
            return min(n_states, int(self.inner_chunk_size))

        if self.parallel_chunk_size is not None:
            return min(n_states, int(self.parallel_chunk_size))

        # Adaptive default:
        # keep enough work per task to amortize executor overhead, but still
        # create multiple tasks per worker for load balancing.
        target_tasks = max(1, int(self.n_workers) * 4)
        adaptive = int(np.ceil(float(n_states) / float(target_tasks)))
        adaptive = max(adaptive, int(self.inner_chunk_size))

        # Align to inner_chunk_size multiple so memory footprint is predictable.
        unit = int(self.inner_chunk_size)
        if unit > 1:
            adaptive = int(np.ceil(float(adaptive) / float(unit)) * unit)
        return min(n_states, max(1, adaptive))

    def estimate_continuation_t1(
        self,
        spot_t1: np.ndarray,
        t_index: int,
        path_prefix: np.ndarray | None = None,
        per_path_r: np.ndarray | float | None = None,
        per_path_sigma: np.ndarray | float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if self.context is None:
            raise ValueError("Provider must be prepared before estimation.")
        call_start = time.perf_counter()

        spot = np.asarray(spot_t1, dtype=np.float64)
        if spot.ndim != 2:
            raise ValueError(f"spot_t1 must have shape (batch, n_outer). Got {spot.shape}")
        batch, n_outer = int(spot.shape[0]), int(spot.shape[1])

        if path_prefix is None:
            prefix = np.zeros((batch, int(t_index) + 1), dtype=np.float64)
            if prefix.shape[1] > 0:
                prefix[:, -1] = np.asarray(self.context.instrument.S0, dtype=np.float64)
        else:
            prefix = np.asarray(path_prefix, dtype=np.float64)
            if prefix.shape != (batch, int(t_index) + 1):
                raise ValueError(
                    "path_prefix shape mismatch. "
                    f"Expected {(batch, int(t_index) + 1)}, got {prefix.shape}."
                )

        r_vec = _resolve_vector(per_path_r, batch=batch, default=float(self.context.instrument.r))
        sigma_vec = np.maximum(
            _resolve_vector(per_path_sigma, batch=batch, default=float(self.context.instrument.sigma)),
            1e-8,
        )

        state_spot = spot.reshape(-1)
        state_prefix = np.repeat(prefix, n_outer, axis=0)
        state_r = np.repeat(r_vec, n_outer)
        state_sigma = np.repeat(sigma_vec, n_outer)

        n_states = int(state_spot.shape[0])
        base_seed = int(self.context.random_seed or 0) if seed is None else int(seed)
        state_chunk_size = self._resolve_state_chunk_size(n_states)

        chunks: list[tuple[int, int]] = []
        step = int(state_chunk_size)
        for start in range(0, n_states, step):
            end = min(start + step, n_states)
            chunks.append((start, end))

        out = np.empty((n_states,), dtype=np.float64)

        def _run_chunk(pair: tuple[int, int]) -> tuple[int, int, np.ndarray]:
            start, end = pair
            vals = self._estimate_states_chunk(
                state_spot=state_spot[start:end],
                state_prefix=state_prefix[start:end],
                state_r=state_r[start:end],
                state_sigma=state_sigma[start:end],
                t_index=int(t_index),
                seed=int(base_seed + start * 1_000_003 + int(t_index) * 97),
            )
            return start, end, vals

        use_parallel = self.parallel_enabled and self.n_workers > 1 and len(chunks) > 1
        if self.verbose:
            self._mc_log(
                f"t={int(t_index)+1}/{int(self.context.n_steps)} start: "
                f"states={n_states}, outer={n_outer}, inner={self.inner_paths}, "
                f"chunks={len(chunks)}, state_chunk_size={state_chunk_size}, "
                f"inner_chunk_size={self.inner_chunk_size}, "
                f"parallel={use_parallel}, workers={self.n_workers}"
            )
        done_chunks = 0
        total_chunks = len(chunks)
        if use_parallel:
            # Keep thread backend for stateful/provider-local closures.
            # Process backend is accepted by config but downgraded here for safety.
            executor_cls: type[concurrent.futures.Executor] = concurrent.futures.ThreadPoolExecutor
            with executor_cls(max_workers=self.n_workers) as ex:
                chunk_iter = iter(chunks)
                inflight: set[concurrent.futures.Future] = set()
                max_inflight = max(self.n_workers * 2, 1)

                def _submit_more():
                    while len(inflight) < max_inflight:
                        try:
                            c = next(chunk_iter)
                        except StopIteration:
                            break
                        inflight.add(ex.submit(_run_chunk, c))

                _submit_more()
                while inflight:
                    done, inflight = concurrent.futures.wait(
                        inflight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done:
                        start, end, vals = fut.result()
                        out[start:end] = vals
                        done_chunks += 1
                        if self.verbose and self.log_every_chunks > 0:
                            if (done_chunks % self.log_every_chunks == 0) or (done_chunks == total_chunks):
                                elapsed = time.perf_counter() - call_start
                                avg_chunk = elapsed / float(done_chunks)
                                eta = avg_chunk * float(total_chunks - done_chunks)
                                self._mc_log(
                                    f"t={int(t_index)+1}: chunk progress {done_chunks}/{total_chunks}, "
                                    f"elapsed={elapsed:.2f}s, eta={eta:.2f}s"
                                )
                    _submit_more()
        else:
            for c in chunks:
                start, end, vals = _run_chunk(c)
                out[start:end] = vals
                done_chunks += 1
                if self.verbose and self.log_every_chunks > 0:
                    if (done_chunks % self.log_every_chunks == 0) or (done_chunks == total_chunks):
                        elapsed = time.perf_counter() - call_start
                        avg_chunk = elapsed / float(done_chunks)
                        eta = avg_chunk * float(total_chunks - done_chunks)
                        self._mc_log(
                            f"t={int(t_index)+1}: chunk progress {done_chunks}/{total_chunks}, "
                            f"elapsed={elapsed:.2f}s, eta={eta:.2f}s"
                        )

        if self.verbose:
            elapsed = time.perf_counter() - call_start
            rate = float(n_states) / elapsed if elapsed > 0 else np.nan
            rate_display = f"{rate:.1f} states/s" if np.isfinite(rate) else "n/a"
            self._mc_log(
                f"t={int(t_index)+1}/{int(self.context.n_steps)} done: "
                f"elapsed={elapsed:.2f}s, rate={rate_display}"
            )
        return out.reshape(batch, n_outer).astype(np.float32)

    def get_price_batch(
        self,
        path_s0: np.ndarray,
        path_r: np.ndarray | float,
        path_sigma: np.ndarray | float,
    ) -> np.ndarray:
        if self.context is None:
            raise ValueError("Provider must be prepared before pricing.")

        s0 = np.asarray(path_s0, dtype=np.float64).reshape(-1)
        batch = int(s0.shape[0])
        r_vec = _resolve_vector(path_r, batch=batch, default=float(self.context.instrument.r))
        sigma_vec = np.maximum(
            _resolve_vector(path_sigma, batch=batch, default=float(self.context.instrument.sigma)),
            1e-8,
        )
        n_steps = int(self.context.n_steps)
        dt = float(self.context.dt)
        claim = self.context.claim

        rng = np.random.default_rng(int(self.context.random_seed or 0) + 11_111)
        segment = _simulate_gbm_segment(
            s_start=s0,
            r_vec=r_vec,
            sigma_vec=sigma_vec,
            dt=dt,
            n_steps=n_steps,
            rng=rng,
            n_scenarios=self.inner_paths,
            use_antithetic=self.use_antithetic,
        )
        full = segment.reshape(-1, n_steps + 1).astype(np.float32)
        payoff = claim.calculate_payoff(tf.convert_to_tensor(full, dtype=tf.float32))
        payoff_np = np.asarray(payoff.numpy(), dtype=np.float64).reshape(batch, self.inner_paths)
        disc = np.exp(-r_vec * float(self.context.maturity))
        return (disc[:, None] * payoff_np).mean(axis=1).astype(np.float32)


class LSMContinuationProvider(ContinuationValueProvider):
    provider_name = "lsm"

    def __init__(
        self,
        train_paths: int = 50_000,
        ridge_alpha: float = 1e-6,
        feature_set: str = "default",
        poly_degree: int = 2,
        use_cache: bool = True,
        cache_dir: str | None = None,
        cache_key: str | None = None,
        force_rebuild: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.train_paths = max(2000, int(train_paths))
        self.ridge_alpha = float(max(ridge_alpha, 1e-12))
        self.feature_set = str(feature_set).strip().lower()
        if self.feature_set not in {"minimal", "default"}:
            raise ValueError("benchmark_lrm_lsm_feature_set must be 'minimal' or 'default'.")
        self.poly_degree = max(1, int(poly_degree))
        self.use_cache = bool(use_cache)
        self.force_rebuild = bool(force_rebuild)
        self.verbose = bool(verbose)
        self.cache_key = None if cache_key is None else str(cache_key).strip()
        self.cache_dir = (
            os.path.normpath(str(cache_dir))
            if cache_dir is not None and str(cache_dir).strip()
            else os.path.normpath(os.path.join(os.getcwd(), "cache", "lrm_lsm"))
        )
        self._cache_file_path: str | None = None
        self.model: _LsmModel | None = None

    def _lsm_log(self, message: str):
        if not self.verbose:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [lrm][lsm] {message}")

    def _build_cache_identity(self) -> dict:
        if self.context is None:
            raise ValueError("Provider must be prepared before building cache identity.")
        claim = self.context.claim
        instrument = self.context.instrument
        identity = {
            "version": 3,
            "provider": self.provider_name,
            "train_paths": int(self.train_paths),
            "ridge_alpha": float(self.ridge_alpha),
            "feature_set": str(self.feature_set),
            "poly_degree": int(self.poly_degree),
            "n_steps": int(self.context.n_steps),
            "dt": float(self.context.dt),
            "maturity": float(self.context.maturity),
            "random_seed": None if self.context.random_seed is None else int(self.context.random_seed),
            "claim_class": type(claim).__name__,
            "claim_option_type": str(getattr(claim, "option_type", "")),
            "claim_strike": float(getattr(claim, "strike", self.context.strike or np.nan)),
            "claim_underlying_index": int(getattr(claim, "underlying_index", 0)),
            "claim_fixing_indices": list(getattr(claim, "fixing_indices", []) or []),
            "instrument_class": type(instrument).__name__,
            "instrument_n": int(getattr(instrument, "N", self.context.n_steps)),
            "instrument_t": float(getattr(instrument, "T", self.context.maturity)),
            "instrument_s0": float(getattr(instrument, "S0", np.nan)),
            "instrument_r": float(getattr(instrument, "r", np.nan)),
            "instrument_sigma": float(getattr(instrument, "sigma", np.nan)),
        }
        return identity

    def _claim_kind(self) -> str:
        if self.context is None:
            return "unknown"
        claim = self.context.claim
        if isinstance(claim, (AsianArithmeticCall, AsianArithmeticPut)):
            return "asian_arithmetic"
        if isinstance(claim, (AsianGeometricCall, AsianGeometricPut)):
            return "asian_geometric"
        return "other"

    def _resolve_fixing_mask(self, n_steps_plus_one: int) -> tuple[np.ndarray, np.ndarray]:
        if self.context is None:
            raise ValueError("Provider must be prepared before resolving fixing mask.")
        claim = self.context.claim
        if hasattr(claim, "resolve_fixing_indices"):
            fixings = np.asarray(claim.resolve_fixing_indices(n_steps_plus_one), dtype=np.int32).reshape(-1)
        else:
            fixings = np.arange(n_steps_plus_one, dtype=np.int32)
        if fixings.size == 0:
            # Defensive fallback: keep one fixing at maturity.
            fixings = np.array([n_steps_plus_one - 1], dtype=np.int32)
        mask = np.zeros((n_steps_plus_one,), dtype=bool)
        mask[fixings] = True
        return fixings, mask

    def _asian_features_training_step(
        self,
        paths: np.ndarray,
        j: int,
        strike: float,
        option_type: str,
        fixing_mask: np.ndarray,
        fixing_counts: np.ndarray,
        cum_fix_sum: np.ndarray,
        cum_fix_log_sum: np.ndarray,
    ) -> list[np.ndarray]:
        if self.context is None:
            return []
        kind = self._claim_kind()
        if kind not in {"asian_arithmetic", "asian_geometric"}:
            return []

        spot_j = paths[:, j]
        total_fixings = max(int(np.sum(fixing_mask)), 1)
        count_j = int(fixing_counts[j])
        count_ratio = np.full_like(spot_j, float(count_j) / float(total_fixings), dtype=np.float64)

        if count_j > 0:
            arith_obs = cum_fix_sum[:, j] / float(count_j)
            geom_obs = np.exp(cum_fix_log_sum[:, j] / float(count_j))
        else:
            # Before first fixing, use spot as neutral placeholder and expose count_ratio=0.
            arith_obs = spot_j.copy()
            geom_obs = spot_j.copy()
        past_log_sum = cum_fix_log_sum[:, j] if count_j > 0 else np.zeros_like(spot_j)
        future_fixing_steps = np.array(
            [idx - int(j) for idx in np.where(fixing_mask)[0] if idx > int(j)],
            dtype=np.int32,
        )
        geo_conditional_price = _geometric_conditional_price_np_batch(
            S=spot_j,
            past_log_sum=past_log_sum,
            total_fixings=total_fixings,
            future_fixing_steps=future_fixing_steps,
            dt=float(self.context.dt),
            r=float(self.context.instrument.r),
            sigma=float(self.context.instrument.sigma),
            strike=float(strike),
            option_type=str(option_type),
        )
        geo_conditional_norm = geo_conditional_price / np.maximum(spot_j, 1e-12)

        if kind == "asian_arithmetic":
            return [
                count_ratio,
                np.log(np.maximum(arith_obs, 1e-12) / float(strike)),
                (arith_obs / np.maximum(spot_j, 1e-12)) - 1.0,
                geo_conditional_norm,
            ]
        return [
            count_ratio,
            np.log(np.maximum(geom_obs, 1e-12) / float(strike)),
            np.log(np.maximum(spot_j, 1e-12) / np.maximum(geom_obs, 1e-12)),
            geo_conditional_norm,
        ]

    def _asian_features_outer_step(
        self,
        path_prefix: np.ndarray | None,
        spot_t1: np.ndarray,
        j: int,
        strike: float,
        n_steps: int,
        option_type: str,
    ) -> list[np.ndarray]:
        kind = self._claim_kind()
        if kind not in {"asian_arithmetic", "asian_geometric"}:
            return []

        spot = np.asarray(spot_t1, dtype=np.float64)
        if spot.ndim != 2:
            raise ValueError(f"spot_t1 must have shape (batch, n_outer). Got {spot.shape}")
        batch, n_outer = int(spot.shape[0]), int(spot.shape[1])
        _, fix_mask = self._resolve_fixing_mask(n_steps + 1)
        total_fixings = max(int(np.sum(fix_mask)), 1)

        if j == 0:
            count_j = 1 if bool(fix_mask[0]) else 0
            count_ratio = np.full((batch, n_outer), float(count_j) / float(total_fixings), dtype=np.float64)
            arith_obs = spot.copy()
            geom_obs = spot.copy()
            if count_j > 0:
                sum_log_j = np.log(np.maximum(spot, 1e-12))
            else:
                sum_log_j = np.zeros_like(spot)
        else:
            if path_prefix is None:
                raise ValueError("path_prefix is required for Asian continuation when t_index >= 0.")
            prefix = np.asarray(path_prefix, dtype=np.float64)
            expected_len = int(j)
            if prefix.shape != (batch, expected_len):
                raise ValueError(
                    "path_prefix shape mismatch for Asian continuation. "
                    f"Expected {(batch, expected_len)}, got {prefix.shape}."
                )

            prefix_mask = fix_mask[:expected_len]
            count_t = int(np.sum(prefix_mask))
            is_fix_j = bool(fix_mask[j]) if j < fix_mask.shape[0] else False

            if count_t > 0:
                sum_fix_t = np.sum(prefix[:, prefix_mask], axis=1)
                sum_log_t = np.sum(np.log(np.maximum(prefix[:, prefix_mask], 1e-12)), axis=1)
            else:
                sum_fix_t = np.zeros((batch,), dtype=np.float64)
                sum_log_t = np.zeros((batch,), dtype=np.float64)

            if is_fix_j:
                count_j = count_t + 1
                sum_fix_j = sum_fix_t[:, None] + spot
                sum_log_j = sum_log_t[:, None] + np.log(np.maximum(spot, 1e-12))
            else:
                count_j = count_t
                sum_fix_j = np.repeat(sum_fix_t[:, None], n_outer, axis=1)
                sum_log_j = np.repeat(sum_log_t[:, None], n_outer, axis=1)

            count_ratio = np.full((batch, n_outer), float(count_j) / float(total_fixings), dtype=np.float64)
            if count_j > 0:
                arith_obs = sum_fix_j / float(count_j)
                geom_obs = np.exp(sum_log_j / float(count_j))
            else:
                arith_obs = spot.copy()
                geom_obs = spot.copy()
                sum_log_j = np.zeros_like(spot)

        future_fixing_steps = np.array(
            [idx - int(j) for idx in np.where(fix_mask)[0] if idx > int(j)],
            dtype=np.int32,
        )
        geo_conditional_price = _geometric_conditional_price_np_batch(
            S=spot.reshape(-1),
            past_log_sum=sum_log_j.reshape(-1),
            total_fixings=total_fixings,
            future_fixing_steps=future_fixing_steps,
            dt=float(self.context.dt),
            r=float(self.context.instrument.r),
            sigma=float(self.context.instrument.sigma),
            strike=float(strike),
            option_type=str(option_type),
        ).reshape(batch, n_outer)
        geo_conditional_norm = geo_conditional_price / np.maximum(spot, 1e-12)

        if kind == "asian_arithmetic":
            return [
                count_ratio,
                np.log(np.maximum(arith_obs, 1e-12) / float(strike)),
                (arith_obs / np.maximum(spot, 1e-12)) - 1.0,
                geo_conditional_norm,
            ]
        return [
            count_ratio,
            np.log(np.maximum(geom_obs, 1e-12) / float(strike)),
            np.log(np.maximum(spot, 1e-12) / np.maximum(geom_obs, 1e-12)),
            geo_conditional_norm,
        ]

    def _resolve_cache_file(self, identity: dict) -> tuple[str, str, str]:
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
        identity_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.cache_key:
            key_hash = hashlib.sha256(self.cache_key.encode("utf-8")).hexdigest()[:24]
            key = f"manual_{key_hash}"
        else:
            key = identity_hash[:24]
        filename = f"lsm_surface_{key}.npz"
        path = os.path.join(self.cache_dir, filename)
        return path, key, identity_hash

    def _try_load_cached_surface(self) -> bool:
        if not self.use_cache or self.force_rebuild:
            return False
        identity = self._build_cache_identity()
        path, key, expected_identity_hash = self._resolve_cache_file(identity)
        self._cache_file_path = path
        if not os.path.isfile(path):
            return False
        try:
            with np.load(path, allow_pickle=False) as npz:
                steps = np.asarray(npz["steps"], dtype=np.int32).reshape(-1)
                beta_matrix = np.asarray(npz["beta_matrix"], dtype=np.float64)
                feature_dim = int(np.asarray(npz["feature_dim"], dtype=np.int32).reshape(-1)[0])
                metadata_raw = np.asarray(npz["metadata_json"]).reshape(-1)
                metadata_json = str(metadata_raw[0]) if metadata_raw.size > 0 else "{}"
            if beta_matrix.ndim != 2:
                raise ValueError(f"Invalid beta_matrix ndim in cache: {beta_matrix.ndim}")
            if beta_matrix.shape[0] != steps.shape[0]:
                raise ValueError(
                    f"Invalid cache: rows mismatch beta_matrix={beta_matrix.shape[0]} vs steps={steps.shape[0]}"
                )
            if beta_matrix.shape[1] != feature_dim:
                raise ValueError(
                    f"Invalid cache: feature_dim mismatch beta_matrix={beta_matrix.shape[1]} vs feature_dim={feature_dim}"
                )

            metadata = json.loads(metadata_json)
            cached_identity_hash = str(metadata.get("identity_hash", ""))
            if cached_identity_hash != expected_identity_hash:
                raise ValueError(
                    "Cached surface identity mismatch with current context/config "
                    f"(cache_key={key})."
                )

            betas_by_step = {
                int(steps[i]): beta_matrix[i].astype(np.float64) for i in range(int(steps.shape[0]))
            }
            self.model = _LsmModel(betas_by_step=betas_by_step, feature_dim=feature_dim)
            self._lsm_log(f"loaded cached surface: {path}")
            return True
        except Exception as exc:
            self._lsm_log(f"failed to load cached surface at '{path}', rebuilding. reason={exc!r}")
            return False

    def _save_cached_surface(self) -> None:
        if not self.use_cache or self.model is None:
            return
        identity = self._build_cache_identity()
        path, key, identity_hash = self._resolve_cache_file(identity)
        self._cache_file_path = path

        os.makedirs(self.cache_dir, exist_ok=True)
        steps = np.array(sorted(self.model.betas_by_step.keys()), dtype=np.int32)
        beta_matrix = np.vstack([self.model.betas_by_step[int(s)] for s in steps]).astype(np.float64)
        feature_dim = np.array([int(self.model.feature_dim)], dtype=np.int32)
        metadata = {
            "provider": self.provider_name,
            "cache_key": key,
            "identity_hash": identity_hash,
            "identity": identity,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        np.savez_compressed(
            path,
            steps=steps,
            beta_matrix=beta_matrix,
            feature_dim=feature_dim,
            metadata_json=np.array([json.dumps(metadata, sort_keys=True)], dtype=np.unicode_),
        )
        self._lsm_log(f"saved cached surface: {path}")

    def _feature_matrix(
        self,
        spot: np.ndarray,
        tau: np.ndarray,
        strike: float,
        running_mean_norm: np.ndarray | None = None,
        realized_vol: np.ndarray | None = None,
        extra_features: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        s = np.maximum(np.asarray(spot, dtype=np.float64).reshape(-1), 1e-12)
        t = np.asarray(tau, dtype=np.float64).reshape(-1)

        cols = [np.ones_like(s), np.log(s), np.log(s / float(strike)), t]
        if self.feature_set == "default":
            if running_mean_norm is None:
                running_mean_norm = np.zeros_like(s)
            if realized_vol is None:
                realized_vol = np.zeros_like(s)
            cols.append(np.asarray(running_mean_norm, dtype=np.float64).reshape(-1))
            cols.append(np.asarray(realized_vol, dtype=np.float64).reshape(-1))
        if extra_features:
            for extra in extra_features:
                cols.append(np.asarray(extra, dtype=np.float64).reshape(-1))

        X_base = np.column_stack(cols)
        if self.poly_degree <= 1:
            return X_base

        poly_cols = [X_base]
        # Keep intercept only once.
        features_no_intercept = X_base[:, 1:]
        for d in range(2, self.poly_degree + 1):
            poly_cols.append(np.power(features_no_intercept, d))
        return np.column_stack(poly_cols)

    def prepare(self, context: ContinuationContext) -> None:
        super().prepare(context)
        if self.context is None:
            raise ValueError("Provider preparation failed.")
        if self._try_load_cached_surface():
            return

        n_steps = int(self.context.n_steps)
        dt = float(self.context.dt)
        claim = self.context.claim
        instrument = self.context.instrument

        sim = instrument.generate_paths(self.train_paths, random_seed=int(self.context.random_seed or 0) + 22_222)
        if isinstance(sim, tuple):
            sim = sim[0]
        paths = np.asarray(tf.convert_to_tensor(sim, dtype=tf.float32).numpy(), dtype=np.float64)
        if paths.ndim != 2 or paths.shape[1] != n_steps + 1:
            raise ValueError(
                "LSM provider currently expects single-instrument paths with shape "
                f"(n_paths, {n_steps + 1}). Got {paths.shape}."
            )

        payoff = claim.calculate_payoff(tf.convert_to_tensor(paths.astype(np.float32), dtype=tf.float32))
        payoff = np.asarray(payoff.numpy(), dtype=np.float64).reshape(-1)
        if payoff.shape[0] != paths.shape[0]:
            raise ValueError("Unexpected payoff shape while fitting LSM provider.")

        log_paths = np.log(np.maximum(paths, 1e-12))
        log_ret = np.diff(log_paths, axis=1)  # (n_paths, n_steps)

        _, fixing_mask = self._resolve_fixing_mask(n_steps + 1)
        fixing_counts = np.cumsum(fixing_mask.astype(np.int32))
        masked_prices = np.where(fixing_mask[None, :], paths, 0.0)
        masked_logs = np.where(fixing_mask[None, :], np.log(np.maximum(paths, 1e-12)), 0.0)
        cum_fix_sum = np.cumsum(masked_prices, axis=1)
        cum_fix_log_sum = np.cumsum(masked_logs, axis=1)

        betas_by_step: dict[int, np.ndarray] = {}
        strike = float(getattr(claim, "strike", self.context.strike or 1.0))
        option_type = str(getattr(claim, "option_type", "call")).strip().lower()
        base_r = float(instrument.r)

        for j in range(0, n_steps + 1):
            tau_j = (n_steps - j) * dt
            y = payoff * np.exp(-base_r * tau_j)

            spot_j = paths[:, j]
            if j == 0:
                running_mean = np.zeros_like(spot_j)
                rv = np.zeros_like(spot_j)
            else:
                prefix = paths[:, : j + 1]
                running_mean = (np.mean(prefix, axis=1) / np.maximum(spot_j, 1e-12)) - 1.0
                ret_pref = log_ret[:, :j]
                rv = np.sqrt(np.maximum(np.var(ret_pref, axis=1), 0.0)) * np.sqrt(252.0)

            asian_extra = self._asian_features_training_step(
                paths=paths,
                j=j,
                strike=strike,
                option_type=option_type,
                fixing_mask=fixing_mask,
                fixing_counts=fixing_counts,
                cum_fix_sum=cum_fix_sum,
                cum_fix_log_sum=cum_fix_log_sum,
            )
            X = self._feature_matrix(
                spot=spot_j,
                tau=np.full_like(spot_j, tau_j, dtype=np.float64),
                strike=strike,
                running_mean_norm=running_mean,
                realized_vol=rv,
                extra_features=asian_extra,
            )
            beta = _fit_ridge(X, y, alpha=self.ridge_alpha)
            betas_by_step[int(j)] = beta

        feature_dim = int(next(iter(betas_by_step.values())).shape[0])
        self.model = _LsmModel(betas_by_step=betas_by_step, feature_dim=feature_dim)
        self._save_cached_surface()

    def _prefix_stats_for_outer(self, path_prefix: np.ndarray, spot_t1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns running_mean_norm and realized_vol for each outer state.
        Shapes: (batch, n_outer), (batch, n_outer)
        """
        batch, n_outer = int(spot_t1.shape[0]), int(spot_t1.shape[1])
        if path_prefix is None or path_prefix.shape[1] == 0:
            zeros = np.zeros((batch, n_outer), dtype=np.float64)
            return zeros, zeros

        prefix = np.asarray(path_prefix, dtype=np.float64)
        pref_len = int(prefix.shape[1])
        pref_sum = prefix.sum(axis=1, keepdims=True)
        running_mean = ((pref_sum + spot_t1) / float(pref_len + 1)) / np.maximum(spot_t1, 1e-12) - 1.0

        if pref_len <= 1:
            rv = np.zeros((batch, n_outer), dtype=np.float64)
            return running_mean, rv

        log_prefix = np.log(np.maximum(prefix, 1e-12))
        ret_prefix = np.diff(log_prefix, axis=1)  # (batch, pref_len-1)
        sum_ret = ret_prefix.sum(axis=1, keepdims=True)
        sum_sq = np.square(ret_prefix).sum(axis=1, keepdims=True)

        s_t = np.maximum(prefix[:, -1:], 1e-12)
        r_new = np.log(np.maximum(spot_t1, 1e-12) / s_t)

        n_ret = float(pref_len)
        sum_all = sum_ret + r_new
        sum_sq_all = sum_sq + np.square(r_new)
        var = np.maximum(sum_sq_all / n_ret - np.square(sum_all / n_ret), 0.0)
        rv = np.sqrt(var) * np.sqrt(252.0)
        return running_mean, rv

    def estimate_continuation_t1(
        self,
        spot_t1: np.ndarray,
        t_index: int,
        path_prefix: np.ndarray | None = None,
        per_path_r: np.ndarray | float | None = None,
        per_path_sigma: np.ndarray | float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        _ = per_path_sigma
        _ = seed
        if self.context is None or self.model is None:
            raise ValueError("LSM provider must be prepared before estimation.")

        spot = np.asarray(spot_t1, dtype=np.float64)
        if spot.ndim != 2:
            raise ValueError(f"spot_t1 must have shape (batch, n_outer). Got {spot.shape}")

        batch, n_outer = int(spot.shape[0]), int(spot.shape[1])
        j = int(t_index) + 1
        n_steps = int(self.context.n_steps)
        tau = max((n_steps - j) * float(self.context.dt), 0.0)

        if j > n_steps:
            raise ValueError(f"Invalid t_index={t_index} for n_steps={n_steps}.")

        beta = self.model.betas_by_step.get(j)
        if beta is None:
            raise ValueError(f"No LSM model found for step j={j}.")

        strike = float(getattr(self.context.claim, "strike", self.context.strike or 1.0))
        option_type = str(getattr(self.context.claim, "option_type", "call")).strip().lower()
        mean_norm, rv = self._prefix_stats_for_outer(path_prefix=path_prefix, spot_t1=spot)
        asian_extra = self._asian_features_outer_step(
            path_prefix=path_prefix,
            spot_t1=spot,
            j=j,
            strike=strike,
            n_steps=n_steps,
            option_type=option_type,
        )

        X = self._feature_matrix(
            spot=spot.reshape(-1),
            tau=np.full((batch * n_outer,), tau, dtype=np.float64),
            strike=strike,
            running_mean_norm=mean_norm.reshape(-1),
            realized_vol=rv.reshape(-1),
            extra_features=[arr.reshape(-1) for arr in asian_extra] if asian_extra else None,
        )
        pred = X @ beta

        # Optional rate adjustment when per-path rates differ from training base rate.
        if per_path_r is not None and tau > 0.0:
            r_vec = _resolve_vector(per_path_r, batch=batch, default=float(self.context.instrument.r))
            base_r = float(self.context.instrument.r)
            adj = np.exp(-(r_vec - base_r)[:, None] * tau)
            pred = pred.reshape(batch, n_outer) * adj
            pred = pred.reshape(-1)

        return np.maximum(pred.reshape(batch, n_outer), 0.0).astype(np.float32)

    def get_price_batch(
        self,
        path_s0: np.ndarray,
        path_r: np.ndarray | float,
        path_sigma: np.ndarray | float,
    ) -> np.ndarray:
        _ = path_sigma
        if self.context is None or self.model is None:
            raise ValueError("LSM provider must be prepared before pricing.")

        s0 = np.asarray(path_s0, dtype=np.float64).reshape(-1)
        batch = int(s0.shape[0])
        r_vec = _resolve_vector(path_r, batch=batch, default=float(self.context.instrument.r))

        strike = float(getattr(self.context.claim, "strike", self.context.strike or 1.0))
        option_type = str(getattr(self.context.claim, "option_type", "call")).strip().lower()
        n_steps = int(self.context.n_steps)
        spot0_2d = s0.reshape(-1, 1)
        asian_extra = self._asian_features_outer_step(
            path_prefix=None,
            spot_t1=spot0_2d,
            j=0,
            strike=strike,
            n_steps=n_steps,
            option_type=option_type,
        )
        X = self._feature_matrix(
            spot=s0,
            tau=np.full((batch,), float(self.context.maturity), dtype=np.float64),
            strike=strike,
            running_mean_norm=np.zeros((batch,), dtype=np.float64),
            realized_vol=np.zeros((batch,), dtype=np.float64),
            extra_features=[arr.reshape(-1) for arr in asian_extra] if asian_extra else None,
        )
        beta0 = self.model.betas_by_step.get(0)
        if beta0 is None:
            raise ValueError("LSM model does not contain step-0 coefficients.")
        pred = X @ beta0

        base_r = float(self.context.instrument.r)
        if np.any(np.abs(r_vec - base_r) > 1e-14):
            pred = pred * np.exp(-(r_vec - base_r) * float(self.context.maturity))
        return np.maximum(pred, 0.0).astype(np.float32)


class AsianMonteCarloContinuationProvider(MonteCarloContinuationProvider):
    """
    MC continuation provider restricted to Asian claims.
    """

    provider_name = "asian_monte_carlo"

    def supports_claim(self, claim) -> bool:
        return isinstance(claim, ASIAN_CLAIMS)


class AsianLSMContinuationProvider(LSMContinuationProvider):
    """
    LSM continuation provider restricted to Asian claims.
    """

    provider_name = "asian_lsmc"

    def supports_claim(self, claim) -> bool:
        return isinstance(claim, ASIAN_CLAIMS)
