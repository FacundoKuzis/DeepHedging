import numpy as np
import tensorflow as tf
import concurrent.futures

from DeepHedging.Agents import DeltaHedgingAgent
from DeepHedging.utils import (
    arithmetic_control_variate_price_delta_crn,
    arithmetic_control_variate_price_delta_crn_batch,
    arithmetic_control_variate_price_delta_crn_batch_worker,
)


class ArithmeticAsianControlVariateAgent(DeltaHedgingAgent):
    """
    Arithmetic Asian benchmark:
    - Conditional MC pricing on remaining fixings
    - Geometric closed-form control variate
    - Delta via CRN central finite differences
    """

    plot_color = "purple"
    name = "arithmetic_asian_control_variate"
    is_trainable = False
    plot_name = {
        "en": "Arithmetic Asian Delta with Control Variate",
        "es": "Delta de Asiatica Aritmetica con Variable de Control",
    }

    def __init__(
        self,
        stock_model,
        option_class,
        num_simulations=10_000,
        bump_size=0.01,
        seed=33,
        no_trade_band=0.0,
        mc_state_chunk_size=64,
        mc_seed_mode="shared_crn",
        parallel_enabled=False,
        n_workers=1,
        parallel_backend="thread",
        parallel_chunk_size=None,
        parallel_min_states=128,
    ):
        super().__init__(stock_model, option_class, no_trade_band=no_trade_band)
        self.option_class = option_class
        self.num_simulations = int(num_simulations)
        self.bump_size = float(bump_size)
        self.seed = int(seed)
        self.mc_state_chunk_size = max(1, int(mc_state_chunk_size))
        self.mc_seed_mode = str(mc_seed_mode).strip().lower()
        if self.mc_seed_mode not in {"shared_crn", "per_state"}:
            raise ValueError("mc_seed_mode must be 'shared_crn' or 'per_state'.")
        self.parallel_enabled = bool(parallel_enabled)
        self.n_workers = max(1, int(n_workers))
        self.parallel_backend = str(parallel_backend).strip().lower()
        if self.parallel_backend not in {"thread", "process"}:
            raise ValueError("parallel_backend must be 'thread' or 'process'.")
        self.parallel_chunk_size = (
            self.mc_state_chunk_size
            if parallel_chunk_size is None
            else max(1, int(parallel_chunk_size))
        )
        self.parallel_min_states = max(1, int(parallel_min_states))

        self.fixing_indices = option_class.resolve_fixing_indices(self.N + 1)
        self.total_fixings = int(len(self.fixing_indices))
        self._running_sum = None
        self._running_log_sum = None

    def build_model(self):
        pass

    def _is_fixing_step(self, step):
        return bool(np.any(self.fixing_indices == int(step)))

    def reset_running_state(self, batch_size):
        self._running_sum = tf.zeros((batch_size,), dtype=tf.float32)
        self._running_log_sum = tf.zeros((batch_size,), dtype=tf.float32)

    def _compute_conditional_deltas_np(self, spot_np, past_sum_np, past_log_sum_np, current_step):
        current_step = int(current_step)
        spot_np = np.asarray(spot_np, dtype=np.float64).reshape(-1)
        past_sum_np = np.asarray(past_sum_np, dtype=np.float64).reshape(-1)
        past_log_sum_np = np.asarray(past_log_sum_np, dtype=np.float64).reshape(-1)

        n_states = int(spot_np.shape[0])
        if n_states == 0:
            return np.empty((0,), dtype=np.float32)

        use_parallel = self.parallel_enabled and self.n_workers > 1 and n_states >= self.parallel_min_states
        chunk_size = self.parallel_chunk_size if use_parallel else self.mc_state_chunk_size
        chunk_size = max(1, int(chunk_size))

        chunk_ranges = []
        for start in range(0, n_states, chunk_size):
            end = min(start + chunk_size, n_states)
            chunk_ranges.append((start, end))

        def _task_kwargs(start, end):
            return {
                "S_t": spot_np[start:end],
                "past_sum": past_sum_np[start:end],
                "past_log_sum": past_log_sum_np[start:end],
                "fixing_indices": self.fixing_indices,
                "current_step": current_step,
                "dt": self.dt,
                "r": self.r,
                "sigma": self.sigma,
                "strike": self.strike,
                "option_type": self.option_type,
                "num_simulations": self.num_simulations,
                "bump_rel": self.bump_size,
                "seed": int(self.seed + 1_000_003 * current_step),
                "seed_mode": self.mc_seed_mode,
                "state_index_offset": int(start),
            }

        deltas_out = np.empty((n_states,), dtype=np.float32)
        if not use_parallel:
            for start, end in chunk_ranges:
                _, deltas_chunk = arithmetic_control_variate_price_delta_crn_batch(**_task_kwargs(start, end))
                deltas_out[start:end] = deltas_chunk.astype(np.float32)
            return deltas_out

        if self.parallel_backend == "process":
            executor_cls = concurrent.futures.ProcessPoolExecutor
            submit_fn = arithmetic_control_variate_price_delta_crn_batch_worker
        else:
            executor_cls = concurrent.futures.ThreadPoolExecutor
            submit_fn = arithmetic_control_variate_price_delta_crn_batch

        with executor_cls(max_workers=self.n_workers) as ex:
            future_to_range = {}
            for start, end in chunk_ranges:
                kwargs = _task_kwargs(start, end)
                if self.parallel_backend == "process":
                    fut = ex.submit(submit_fn, kwargs)
                else:
                    fut = ex.submit(submit_fn, **kwargs)
                future_to_range[fut] = (start, end)

            for fut in concurrent.futures.as_completed(future_to_range):
                start, end = future_to_range[fut]
                _, deltas_chunk = fut.result()
                deltas_out[start:end] = np.asarray(deltas_chunk, dtype=np.float32)
        return deltas_out

    def process_batch(self, batch_paths, batch_T_minus_t):
        batch_size = batch_paths.shape[0]
        self.reset_last_delta(batch_size)
        self.reset_running_state(batch_size)

        all_actions = []
        for t in range(batch_paths.shape[1] - 1):  # Hedge up to T-1
            current_paths = batch_paths[:, t, :]
            current_spot = tf.maximum(tf.cast(current_paths[:, 0], tf.float32), 1e-12)

            # State at time t includes current fixing if t is in fixing calendar.
            if self._is_fixing_step(t):
                self._running_sum += current_spot
                self._running_log_sum += tf.math.log(current_spot)

            target_delta = tf.numpy_function(
                self._compute_conditional_deltas_np,
                [current_spot, self._running_sum, self._running_log_sum, np.int32(t)],
                tf.float32,
            )
            target_delta.set_shape((batch_size,))

            action = self._to_actions(target_delta, current_paths)
            all_actions.append(action)

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_size, 1, all_actions.shape[-1]), dtype=tf.float32)
        all_actions = tf.concat([all_actions, zero_action], axis=1)
        return all_actions

    def get_model_price(self):
        past_sum = float(self.S0) if self._is_fixing_step(0) else 0.0
        past_log_sum = float(np.log(max(self.S0, 1e-12))) if self._is_fixing_step(0) else 0.0
        price, _ = arithmetic_control_variate_price_delta_crn(
            S_t=float(self.S0),
            past_sum=past_sum,
            past_log_sum=past_log_sum,
            fixing_indices=self.fixing_indices,
            current_step=0,
            dt=self.dt,
            r=self.r,
            sigma=self.sigma,
            strike=self.strike,
            option_type=self.option_type,
            num_simulations=self.num_simulations,
            bump_rel=self.bump_size,
            seed=self.seed,
        )
        return float(price)
