import numpy as np
import tensorflow as tf
from DeepHedging.Agents import DeltaHedgingAgent
from DeepHedging.utils import arithmetic_control_variate_price_delta_crn


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
    ):
        super().__init__(stock_model, option_class, no_trade_band=no_trade_band)
        self.option_class = option_class
        self.num_simulations = int(num_simulations)
        self.bump_size = float(bump_size)
        self.seed = int(seed)

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
        deltas = np.zeros_like(spot_np, dtype=np.float32)
        for i, (s_i, ps_i, pls_i) in enumerate(zip(spot_np, past_sum_np, past_log_sum_np)):
            _, delta_i = arithmetic_control_variate_price_delta_crn(
                S_t=float(s_i),
                past_sum=float(ps_i),
                past_log_sum=float(pls_i),
                fixing_indices=self.fixing_indices,
                current_step=current_step,
                dt=self.dt,
                r=self.r,
                sigma=self.sigma,
                strike=self.strike,
                option_type=self.option_type,
                num_simulations=self.num_simulations,
                bump_rel=self.bump_size,
                seed=self.seed + 1_000_003 * current_step + i,
            )
            deltas[i] = np.float32(delta_i)
        return deltas

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

