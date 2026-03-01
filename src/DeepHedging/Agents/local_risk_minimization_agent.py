from __future__ import annotations

from datetime import datetime
import time

import numpy as np
import tensorflow as tf

from DeepHedging.Agents.delta_hedging_agent import DeltaHedgingAgent
from DeepHedging.ContingentClaims import EuropeanCall, EuropeanPut
from DeepHedging.utils.lrm_continuation import ContinuationContext, build_continuation_provider
from DeepHedging.utils.lrm_engine import compute_lrm_target_batch


class LocalRiskMinimizationAgent(DeltaHedgingAgent):
    """
    Local Risk Minimization benchmark with pluggable continuation providers.

    Implements one-step LRM hedge ratio:
      h_t = Cov(C_{t+1}, dS_{t+1}) / Var(dS_{t+1})

    where C_{t+1} is supplied by a continuation provider.
    """

    plot_color = "saddlebrown"
    name = "local_risk_minimization"
    is_trainable = False
    plot_name = {
        "en": "Local Risk Minimization",
        "es": "Agente Local Risk Minimization",
    }

    def __init__(
        self,
        stock_model,
        option_class,
        no_trade_band: float = 0.0,
        no_trade_band_mode: str = "absolute",
        no_trade_band_eps: float = 1e-8,
        # LRM core
        lrm_provider: str = "bs_closed_form",
        lrm_outer_paths: int = 512,
        lrm_var_epsilon: float = 1e-10,
        lrm_use_antithetic: bool = True,
        lrm_seed_mode: str = "shared_crn",
        # MC provider params
        lrm_mc_inner_paths: int = 1024,
        lrm_mc_inner_chunk_size: int = 64,
        lrm_mc_parallel_enabled: bool = False,
        lrm_mc_n_workers: int = 1,
        lrm_mc_parallel_backend: str = "thread",
        lrm_mc_parallel_chunk_size: int | None = None,
        # LSM provider params
        lrm_lsm_train_paths: int = 50_000,
        lrm_lsm_ridge_alpha: float = 1e-6,
        lrm_lsm_feature_set: str = "default",
        lrm_lsm_poly_degree: int = 2,
        lrm_lsm_use_cache: bool = True,
        lrm_lsm_cache_dir: str | None = None,
        lrm_lsm_cache_key: str | None = None,
        lrm_lsm_force_rebuild: bool = False,
        # logging/debug
        lrm_verbose: bool = False,
        lrm_log_every_t: int = 5,
        lrm_mc_log_every_chunks: int = 0,
        # seed
        seed: int = 33,
        **kwargs,
    ):
        _ = kwargs
        super().__init__(
            stock_model,
            option_class,
            no_trade_band=no_trade_band,
            no_trade_band_mode=no_trade_band_mode,
            no_trade_band_eps=no_trade_band_eps,
        )
        self.option_class = option_class

        self.seed = int(seed)
        self.lrm_outer_paths = max(2, int(lrm_outer_paths))
        self.lrm_var_epsilon = float(max(lrm_var_epsilon, 1e-14))
        self.lrm_use_antithetic = bool(lrm_use_antithetic)
        self.lrm_seed_mode = str(lrm_seed_mode).strip().lower()
        self.lrm_verbose = bool(lrm_verbose)
        self.lrm_log_every_t = max(1, int(lrm_log_every_t))
        if self.lrm_seed_mode not in {"shared_crn", "per_state"}:
            raise ValueError("lrm_seed_mode must be 'shared_crn' or 'per_state'.")

        self._enforce_single_underlying_scope(option_class)

        self.continuation_provider = build_continuation_provider(
            provider_name=str(lrm_provider),
            inner_paths=int(lrm_mc_inner_paths),
            inner_chunk_size=int(lrm_mc_inner_chunk_size),
            parallel_enabled=bool(lrm_mc_parallel_enabled),
            n_workers=int(lrm_mc_n_workers),
            parallel_backend=str(lrm_mc_parallel_backend),
            parallel_chunk_size=(
                None if lrm_mc_parallel_chunk_size is None else int(lrm_mc_parallel_chunk_size)
            ),
            use_antithetic=bool(lrm_use_antithetic),
            train_paths=int(lrm_lsm_train_paths),
            ridge_alpha=float(lrm_lsm_ridge_alpha),
            feature_set=str(lrm_lsm_feature_set),
            poly_degree=int(lrm_lsm_poly_degree),
            use_cache=bool(lrm_lsm_use_cache),
            cache_dir=None if lrm_lsm_cache_dir is None else str(lrm_lsm_cache_dir),
            cache_key=None if lrm_lsm_cache_key is None else str(lrm_lsm_cache_key),
            force_rebuild=bool(lrm_lsm_force_rebuild),
            verbose=bool(lrm_verbose),
            log_every_chunks=int(lrm_mc_log_every_chunks),
        )

        ctx = ContinuationContext(
            claim=option_class,
            instrument=stock_model,
            n_steps=int(stock_model.N),
            maturity=float(stock_model.T),
            dt=float(stock_model.dt),
            strike=float(getattr(option_class, "strike", np.nan)),
            option_type=str(getattr(option_class, "option_type", "")),
            random_seed=int(self.seed),
        )
        self.continuation_provider.prepare(ctx)
        if not self.continuation_provider.supports_claim(option_class):
            raise ValueError(
                f"Provider '{self.continuation_provider.provider_name}' does not support claim '{type(option_class).__name__}'."
            )
        if self.lrm_verbose:
            self._lrm_log(
                "initialized "
                f"provider={self.continuation_provider.provider_name}, "
                f"outer_paths={self.lrm_outer_paths}, seed_mode={self.lrm_seed_mode}, "
                f"log_every_t={self.lrm_log_every_t}"
            )

    def _enforce_single_underlying_scope(self, option_class):
        # v1 scope: single underlying paths only.
        if int(self.N) <= 0:
            raise ValueError("N must be > 0.")
        if isinstance(getattr(option_class, "underlying_index", 0), (list, tuple)):
            raise ValueError("LocalRiskMinimizationAgent v1 supports single underlying_index only.")

    def build_model(self):
        pass

    def _lrm_log(self, message: str):
        if not self.lrm_verbose:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [lrm][agent:{self.name}] {message}")

    def _resolve_path_vector(
        self,
        values,
        batch_size: int,
        default: float,
        allow_matrix: bool = False,
    ) -> np.ndarray:
        if values is None:
            return np.full((batch_size,), float(default), dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return np.full((batch_size,), float(arr), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1)
            if arr.shape[0] != int(batch_size):
                raise ValueError(
                    f"Path-wise vector length mismatch: expected {batch_size}, got {arr.shape[0]}."
                )
            return arr
        if allow_matrix and arr.ndim == 2:
            if arr.shape[0] != int(batch_size):
                raise ValueError(
                    f"Path-wise matrix batch mismatch: expected {batch_size}, got {arr.shape[0]}."
                )
            return arr
        raise ValueError(
            f"Unsupported path-wise shape {arr.shape}. Expected scalar, (batch,) or (batch, n_steps)."
        )

    def process_batch(
        self,
        batch_paths,
        batch_T_minus_t,
        batch_path_r=None,
        batch_path_sigma=None,
        batch_history_features=None,
        batch_pre_history_prices=None,
    ):
        _ = batch_history_features
        _ = batch_pre_history_prices
        paths = tf.convert_to_tensor(batch_paths, dtype=tf.float32)
        if len(paths.shape) == 2:
            paths = tf.expand_dims(paths, axis=-1)
        _ = tf.convert_to_tensor(batch_T_minus_t, dtype=tf.float32)

        if len(paths.shape) != 3:
            raise ValueError(f"batch_paths must be rank 3, got shape={paths.shape}")
        if int(paths.shape[2]) != 1:
            raise ValueError(
                "LocalRiskMinimizationAgent currently supports exactly one underlying instrument."
            )

        batch_size = int(paths.shape[0])
        n_steps = int(paths.shape[1]) - 1
        if n_steps != int(self.N):
            raise ValueError(
                f"Path timestep mismatch for LRM: expected N={self.N}, got {n_steps}."
            )

        r_vec = self._resolve_path_vector(
            batch_path_r,
            batch_size=batch_size,
            default=float(self.r),
            allow_matrix=True,
        )
        sigma_vec = np.maximum(
            self._resolve_path_vector(
                batch_path_sigma,
                batch_size=batch_size,
                default=float(self.sigma),
                allow_matrix=True,
            ),
            1e-8,
        )
        if sigma_vec.ndim == 2 and int(sigma_vec.shape[1]) < n_steps:
            raise ValueError(
                f"batch_path_sigma timestep mismatch for LRM: expected at least {n_steps}, got {sigma_vec.shape[1]}."
            )

        self.reset_last_delta(batch_size)
        all_actions = []
        loop_start = time.perf_counter()

        path_np = np.asarray(paths.numpy(), dtype=np.float64)[:, :, 0]
        if self.lrm_verbose:
            self._lrm_log(
                f"process_batch start: batch_size={batch_size}, n_steps={n_steps}, "
                f"provider={self.continuation_provider.provider_name}"
            )
        for t in range(n_steps):
            step_start = time.perf_counter()
            spot_t = path_np[:, t]
            prefix_t = path_np[:, : t + 1]
            r_t = r_vec[:, t] if isinstance(r_vec, np.ndarray) and r_vec.ndim == 2 else r_vec
            sigma_t = sigma_vec[:, t] if sigma_vec.ndim == 2 else sigma_vec

            target_delta = compute_lrm_target_batch(
                spot_t=spot_t,
                t_index=t,
                dt=float(self.dt),
                provider=self.continuation_provider,
                per_path_r=r_t,
                per_path_sigma=sigma_t,
                path_prefix=prefix_t,
                outer_paths=int(self.lrm_outer_paths),
                var_epsilon=float(self.lrm_var_epsilon),
                use_antithetic=bool(self.lrm_use_antithetic),
                seed=int(self.seed + 10_003 * t),
                seed_mode=self.lrm_seed_mode,
            )

            current_paths = paths[:, t, :]  # (batch, n_instruments)
            action = self._to_actions(tf.convert_to_tensor(target_delta, dtype=tf.float32), current_paths)
            all_actions.append(action)

            if self.lrm_verbose:
                done = t + 1
                should_log = (done == 1) or (done == n_steps) or (done % self.lrm_log_every_t == 0)
                if should_log:
                    now = time.perf_counter()
                    step_elapsed = now - step_start
                    elapsed = now - loop_start
                    avg_step = elapsed / float(done)
                    eta = avg_step * float(n_steps - done)
                    self._lrm_log(
                        f"t={done}/{n_steps} step_elapsed={step_elapsed:.3f}s "
                        f"elapsed={elapsed:.2f}s eta={eta:.2f}s"
                    )

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_size, 1, all_actions.shape[-1]), dtype=tf.float32)
        return tf.concat([all_actions, zero_action], axis=1)

    def get_model_price(self):
        # Scalar fallback for compatibility.
        price = self.get_model_price_batch(
            path_s0=np.array([float(self.S0)], dtype=np.float32),
            path_r=np.array([float(self.r)], dtype=np.float32),
            path_sigma=np.array([float(self.sigma)], dtype=np.float32),
        )
        return float(np.asarray(price, dtype=np.float64).reshape(-1)[0])

    def get_model_price_batch(self, path_s0=None, path_r=None, path_sigma=None):
        if path_s0 is None:
            s0 = np.array([float(self.S0)], dtype=np.float32)
        else:
            s0 = np.asarray(path_s0, dtype=np.float32).reshape(-1)
        n = int(s0.shape[0])

        r = self._resolve_path_vector(
            path_r,
            batch_size=n,
            default=float(self.r),
            allow_matrix=True,
        )
        if isinstance(r, np.ndarray) and r.ndim == 2:
            if int(r.shape[1]) < 1:
                raise ValueError("path_r matrix must have at least one column.")
            r = r[:, 0]
        sigma_resolved = self._resolve_path_vector(
            path_sigma,
            batch_size=n,
            default=float(self.sigma),
            allow_matrix=True,
        )
        # For pricing at t0, if caller provides stepwise sigma surface (batch, n_steps),
        # use the first decision-time sigma per path.
        if isinstance(sigma_resolved, np.ndarray) and sigma_resolved.ndim == 2:
            if int(sigma_resolved.shape[1]) < 1:
                raise ValueError("path_sigma matrix must have at least one column.")
            sigma_resolved = sigma_resolved[:, 0]
        sigma = np.maximum(np.asarray(sigma_resolved, dtype=np.float64), 1e-8)

        provider = self.continuation_provider
        if isinstance(provider, type(None)):
            raise ValueError("Continuation provider is not initialized.")

        if hasattr(provider, "get_price_batch"):
            prices = provider.get_price_batch(path_s0=s0, path_r=r, path_sigma=sigma)
            return tf.convert_to_tensor(np.asarray(prices, dtype=np.float32).reshape(-1), dtype=tf.float32)

        # Fallback: BS closed-form for european claims only.
        if not isinstance(self.option_class, (EuropeanCall, EuropeanPut)):
            raise ValueError("Pathwise price batch unavailable for this provider/claim combination.")
        from DeepHedging.utils.lrm_providers import _bs_price_batch  # local import

        option_type = str(getattr(self.option_class, "option_type", "call")).strip().lower()
        tau = np.full((n,), float(self.T), dtype=np.float64)
        prices = _bs_price_batch(
            option_type=option_type,
            spot=s0.astype(np.float64),
            strike=float(self.strike),
            rate=r.astype(np.float64),
            sigma=sigma.astype(np.float64),
            tau=tau,
        )
        return tf.convert_to_tensor(prices.astype(np.float32), dtype=tf.float32)
