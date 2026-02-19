import numpy as np
import tensorflow as tf
from DeepHedging.Agents import DeltaHedgingAgent
from DeepHedging.utils import (
    geometric_conditional_delta_bump_tf,
    geometric_conditional_price_tf,
)


class GeometricAsianDeltaHedgingAgent(DeltaHedgingAgent):
    """
    Delta benchmark for discrete geometric Asian options under GBM.

    Uses conditional pricing/delta based on observed fixings up to time t and
    remaining fixing schedule for t+1..T.
    """

    plot_color = "orange"
    name = "asian_delta_hedging"
    is_trainable = False
    plot_name = {
        "en": "Geometric Asian Delta",
        "es": "Delta de Opcion Asiatica Geometrica",
    }

    def __init__(self, stock_model, option_class, bump_size=0.01, no_trade_band=0.0):
        super().__init__(stock_model, option_class, no_trade_band=no_trade_band)
        self.bump_size = float(bump_size)
        self.fixing_indices = option_class.resolve_fixing_indices(self.N + 1)
        self.total_fixings = int(len(self.fixing_indices))
        self._running_log_sum = None

    def _future_fixing_steps(self, current_step):
        return np.array(
            [idx - current_step for idx in self.fixing_indices if idx > current_step],
            dtype=np.int32,
        )

    def _is_fixing_step(self, step):
        return bool(np.any(self.fixing_indices == int(step)))

    def reset_running_state(self, batch_size):
        self._running_log_sum = tf.zeros((batch_size,), dtype=tf.float32)

    def delta(self, S, T_minus_t):
        """
        Backward-compatible delta signature.
        For process_batch(), a stateful conditional delta is used instead.
        """
        S = tf.convert_to_tensor(S, dtype=tf.float32)
        T_minus_t = tf.convert_to_tensor(T_minus_t, dtype=tf.float32)

        steps_remaining = tf.maximum(
            tf.cast(tf.round(T_minus_t / tf.constant(self.dt, dtype=tf.float32)), tf.int32),
            0,
        )

        def _single_delta(args):
            s_i, n_i = args
            n_i = int(n_i)
            future_steps = np.arange(1, n_i + 1, dtype=np.int32)
            d_i = geometric_conditional_delta_bump_tf(
                S=tf.reshape(s_i, (1,)),
                past_log_sum=tf.zeros((1,), dtype=tf.float32),
                total_fixings=max(n_i, 1),
                future_fixing_steps=future_steps,
                dt=self.dt,
                r=self.r,
                sigma=self.sigma,
                strike=self.strike,
                option_type=self.option_type,
                bump_rel=self.bump_size,
            )[0]
            return d_i

        return tf.map_fn(
            _single_delta,
            (S, steps_remaining),
            fn_output_signature=tf.float32,
        )

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
                self._running_log_sum += tf.math.log(current_spot)

            future_fixings = self._future_fixing_steps(t)
            target_delta = geometric_conditional_delta_bump_tf(
                S=current_spot,
                past_log_sum=self._running_log_sum,
                total_fixings=self.total_fixings,
                future_fixing_steps=future_fixings,
                dt=self.dt,
                r=self.r,
                sigma=self.sigma,
                strike=self.strike,
                option_type=self.option_type,
                bump_rel=self.bump_size,
            )

            action = self._to_actions(target_delta, current_paths)
            all_actions.append(action)

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_size, 1, all_actions.shape[-1]), dtype=tf.float32)
        all_actions = tf.concat([all_actions, zero_action], axis=1)
        return all_actions

    def get_model_price(self):
        spot = tf.constant([float(self.S0)], dtype=tf.float32)
        past_log_sum = tf.zeros((1,), dtype=tf.float32)
        if self._is_fixing_step(0):
            past_log_sum += tf.math.log(tf.maximum(spot, 1e-12))

        future_fixings = self._future_fixing_steps(0)
        price = geometric_conditional_price_tf(
            S=spot,
            past_log_sum=past_log_sum,
            total_fixings=self.total_fixings,
            future_fixing_steps=future_fixings,
            dt=self.dt,
            r=self.r,
            sigma=self.sigma,
            strike=self.strike,
            option_type=self.option_type,
        )
        return price[0]

