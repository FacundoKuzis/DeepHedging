import concurrent.futures
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from DeepHedging.HedgingInstruments import GBMStock, HestonStock


class MonteCarloPricer:
    """
    Monte Carlo option pricer with finite-difference delta.

    Backward compatible API + optional batch/parallel delta evaluation.
    """

    def __init__(
        self,
        stock_model,
        r,
        T,
        num_simulations=10_000,
        seed=None,
        use_vectorized=True,
        chunk_size=64,
        parallel_enabled=False,
        n_workers=1,
        parallel_backend="thread",
    ):
        self.stock_model = stock_model
        self.r = float(r)
        self.T = float(T)
        self.num_simulations = int(num_simulations)
        self.seed = None if seed is None else int(seed)

        self.use_vectorized = bool(use_vectorized)
        self.chunk_size = max(1, int(chunk_size))
        self.parallel_enabled = bool(parallel_enabled)
        self.n_workers = max(1, int(n_workers))
        self.parallel_backend = str(parallel_backend).strip().lower()
        if self.parallel_backend not in {"thread", "process"}:
            raise ValueError("parallel_backend must be 'thread' or 'process'.")

    def _discount_factor(self, T=None, r=None):
        T_eff = float(self.T if T is None else T)
        r_eff = float(self.r if r is None else r)
        return float(np.exp(-r_eff * T_eff))

    def _simulate_gbm_paths_from_dW(self, S0, dW, T=None, r=None, sigma=None):
        """
        Fast vectorized GBM path generator given Brownian increments dW.
        """
        T_eff = float(self.T if T is None else T)
        r_eff = float(self.stock_model.r if r is None else r)
        sigma_eff = float(self.stock_model.sigma if sigma is None else sigma)
        n_steps = int(self.stock_model.N)
        dt = float(T_eff / n_steps)

        drift = (r_eff - 0.5 * sigma_eff**2) * dt
        increments = drift + sigma_eff * dW
        log_cum = np.cumsum(increments, axis=1)
        log_full = np.concatenate(
            [np.zeros((log_cum.shape[0], 1), dtype=np.float64), log_cum],
            axis=1,
        )
        paths = float(S0) * np.exp(log_full)
        return tf.convert_to_tensor(paths, dtype=tf.float32)

    def _draw_gbm_dW(self, seed=None, T=None):
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        T_eff = float(self.T if T is None else T)
        n_steps = int(self.stock_model.N)
        dt = float(T_eff / n_steps)
        return rng.normal(
            loc=0.0,
            scale=np.sqrt(dt),
            size=(self.num_simulations, int(self.stock_model.N)),
        )

    def simulate_paths(self, seed=None, S0=None, dW=None, T=None):
        """
        Simulates asset paths using stock_model. For GBM, allows reusing dW across bumps.
        """
        if isinstance(self.stock_model, GBMStock):
            T_eff = float(self.T if T is None else T)
            if dW is None:
                dW = self._draw_gbm_dW(seed=seed, T=T_eff)
            s0 = float(self.stock_model.S0 if S0 is None else S0)
            return self._simulate_gbm_paths_from_dW(s0, dW, T=T_eff)

        # Fallback for non-GBM models.
        original_S0 = getattr(self.stock_model, "S0", None)
        original_T = getattr(self.stock_model, "T", None)
        original_dt = getattr(self.stock_model, "dt", None)
        if S0 is not None and original_S0 is not None:
            self.stock_model.S0 = float(S0)
        if T is not None and original_T is not None:
            self.stock_model.T = float(T)
            if hasattr(self.stock_model, "N"):
                self.stock_model.dt = float(T) / float(self.stock_model.N)
        try:
            draw_seed = self.seed if seed is None else int(seed)
            if isinstance(self.stock_model, HestonStock):
                S_paths, _ = self.stock_model.generate_paths(
                    num_paths=self.num_simulations,
                    random_seed=draw_seed,
                )
                return S_paths
            return self.stock_model.generate_paths(
                num_paths=self.num_simulations,
                random_seed=draw_seed,
            )
        finally:
            if S0 is not None and original_S0 is not None:
                self.stock_model.S0 = original_S0
            if T is not None and original_T is not None:
                self.stock_model.T = original_T
                if original_dt is not None:
                    self.stock_model.dt = original_dt

    def price(self, contingent_claim, paths=None, T=None, r=None):
        T_eff = self.T if T is None else float(T)
        r_eff = self.r if r is None else float(r)
        if paths is None:
            paths = self.simulate_paths(T=T_eff)
        payoffs = contingent_claim.calculate_payoff(paths)
        payoffs = tf.cast(payoffs, tf.float32)
        discounted_payoff = self._discount_factor(T=T_eff, r=r_eff) * tf.reduce_mean(payoffs)
        return float(discounted_payoff.numpy())

    def delta(self, contingent_claim, bump_size=0.01, use_common_random_numbers=True, seed=None):
        base_S0 = float(self.stock_model.S0)
        epsilon = max(abs(base_S0) * float(bump_size), 1e-8)
        return self.delta_with_S0(
            contingent_claim=contingent_claim,
            S0=base_S0,
            bump_size=bump_size,
            use_common_random_numbers=use_common_random_numbers,
            seed=seed,
            epsilon=epsilon,
        )

    def delta_with_S0(
        self,
        contingent_claim,
        S0,
        bump_size=0.01,
        use_common_random_numbers=True,
        seed=None,
        epsilon=None,
        T=None,
        r=None,
    ):
        s0 = float(S0)
        T_eff = self.T if T is None else float(T)
        r_eff = self.r if r is None else float(r)
        eps = max(abs(s0) * float(bump_size), 1e-8) if epsilon is None else float(epsilon)
        up_s0 = s0 + eps
        down_s0 = max(s0 - eps, 1e-8)

        if isinstance(self.stock_model, GBMStock):
            if use_common_random_numbers:
                dW = self._draw_gbm_dW(seed=seed, T=T_eff)
                paths_up = self.simulate_paths(S0=up_s0, dW=dW, T=T_eff)
                paths_down = self.simulate_paths(S0=down_s0, dW=dW, T=T_eff)
            else:
                paths_up = self.simulate_paths(S0=up_s0, seed=seed, T=T_eff)
                down_seed = None if seed is None else int(seed) + 1_000_003
                paths_down = self.simulate_paths(S0=down_s0, seed=down_seed, T=T_eff)
            price_up = self.price(contingent_claim, paths=paths_up, T=T_eff, r=r_eff)
            price_down = self.price(contingent_claim, paths=paths_down, T=T_eff, r=r_eff)
            return float((price_up - price_down) / (2.0 * eps))

        # Fallback for non-GBM models.
        price_up = self.price_with_S0(contingent_claim, up_s0, T=T_eff, r=r_eff)
        price_down = self.price_with_S0(contingent_claim, down_s0, T=T_eff, r=r_eff)
        return float((price_up - price_down) / (2.0 * eps))

    def price_with_S0(self, contingent_claim, S0, paths=None, T=None, r=None):
        T_eff = self.T if T is None else float(T)
        r_eff = self.r if r is None else float(r)
        if paths is None:
            paths = self.simulate_paths(S0=float(S0), T=T_eff)
        payoffs = contingent_claim.calculate_payoff(paths)
        payoffs = tf.cast(payoffs, tf.float32)
        discounted_payoff = self._discount_factor(T=T_eff, r=r_eff) * tf.reduce_mean(payoffs)
        return float(discounted_payoff.numpy())

    def delta_batch_with_S0(
        self,
        contingent_claim,
        S0_values,
        T_values=None,
        bump_size=0.01,
        use_common_random_numbers=True,
        seed=None,
        chunk_size=None,
        parallel_enabled=None,
        n_workers=None,
        parallel_backend=None,
        progress_callback=None,
        profile=False,
    ):
        """
        Batch delta computation with chunking and optional parallel execution.
        """
        t_start = time.perf_counter()

        s_arr = np.asarray(S0_values, dtype=np.float64).reshape(-1)
        if s_arr.size == 0:
            return np.empty((0,), dtype=np.float32)

        if T_values is None:
            t_arr = np.full((s_arr.size,), float(self.T), dtype=np.float64)
        else:
            t_arr = np.asarray(T_values, dtype=np.float64).reshape(-1)
            if t_arr.size == 1:
                t_arr = np.full((s_arr.size,), float(t_arr[0]), dtype=np.float64)
            if t_arr.size != s_arr.size:
                raise ValueError("T_values length mismatch with S0_values.")

        cs = self.chunk_size if chunk_size is None else max(1, int(chunk_size))
        pe = self.parallel_enabled if parallel_enabled is None else bool(parallel_enabled)
        workers = self.n_workers if n_workers is None else max(1, int(n_workers))
        backend = self.parallel_backend if parallel_backend is None else str(parallel_backend).strip().lower()

        if backend not in {"thread", "process"}:
            raise ValueError("parallel_backend must be 'thread' or 'process'.")
        if backend == "process":
            warnings.warn(
                "Process backend is not supported safely for MonteCarloPricer stateful objects; using thread backend.",
                RuntimeWarning,
            )
            backend = "thread"

        base_seed = self.seed if seed is None else int(seed)
        if base_seed is None:
            base_seed = 0

        chunk_ranges = []
        for start in range(0, s_arr.size, cs):
            end = min(start + cs, s_arr.size)
            chunk_ranges.append((start, end))

        def _compute_chunk(start_end):
            start, end = start_end
            out = np.empty((end - start,), dtype=np.float32)
            for local_i, i in enumerate(range(start, end)):
                state_seed = int(base_seed + i * 1_000_003)
                out[local_i] = np.float32(
                    self.delta_with_S0(
                        contingent_claim=contingent_claim,
                        S0=float(s_arr[i]),
                        T=float(t_arr[i]),
                        bump_size=float(bump_size),
                        use_common_random_numbers=bool(use_common_random_numbers),
                        seed=state_seed,
                    )
                )
            return start, end, out

        deltas = np.empty((s_arr.size,), dtype=np.float32)

        if pe and workers > 1 and len(chunk_ranges) > 1:
            executor_cls = concurrent.futures.ThreadPoolExecutor
            with executor_cls(max_workers=workers) as ex:
                futures = [ex.submit(_compute_chunk, cr) for cr in chunk_ranges]
                completed = 0
                for fut in concurrent.futures.as_completed(futures):
                    start, end, vals = fut.result()
                    deltas[start:end] = vals
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, len(chunk_ranges), end, s_arr.size)
        else:
            for c_idx, cr in enumerate(chunk_ranges, start=1):
                start, end, vals = _compute_chunk(cr)
                deltas[start:end] = vals
                if progress_callback is not None:
                    progress_callback(c_idx, len(chunk_ranges), end, s_arr.size)

        if not profile:
            return deltas

        elapsed = time.perf_counter() - t_start
        profile_dict = {
            "n_states": int(s_arr.size),
            "num_simulations": int(self.num_simulations),
            "chunk_size": int(cs),
            "parallel_enabled": bool(pe),
            "n_workers": int(workers),
            "parallel_backend": str(backend),
            "elapsed_seconds": float(elapsed),
        }
        return deltas, profile_dict

    @staticmethod
    def save_profile_csv(profile_rows, output_csv: str):
        os_dir = os.path.dirname(output_csv)
        if os_dir:
            import os

            os.makedirs(os_dir, exist_ok=True)
        df = pd.DataFrame(profile_rows)
        df.to_csv(output_csv, index=False)
        return output_csv
