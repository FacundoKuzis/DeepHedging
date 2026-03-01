import numpy as np
import tensorflow as tf
from DeepHedging.Agents import DeltaHedgingAgent
from DeepHedging.utils import (
    geometric_conditional_delta_bump_tf,
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

    def _normalize_rate_vector(self, path_r, n_paths):
        rate = self._normalize_rate_input(self.r if path_r is None else path_r, n_paths)
        rate = tf.convert_to_tensor(rate, dtype=tf.float32)
        if rate.shape.rank == 0:
            rate = tf.fill((n_paths,), rate)
        else:
            rate = tf.reshape(rate, (-1,))
        return rate

    def _normalize_sigma_vector(self, path_sigma, n_paths, n_steps=None):
        sigma = self._normalize_sigma_input(
            self.sigma if path_sigma is None else path_sigma,
            n_paths,
            n_steps=n_steps,
        )
        sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
        if sigma.shape.rank == 0:
            sigma = tf.fill((n_paths,), sigma)
        elif sigma.shape.rank == 1:
            sigma = tf.reshape(sigma, (-1,))
        elif sigma.shape.rank == 2:
            sigma = tf.reshape(sigma, (n_paths, -1))
        else:
            raise ValueError(
                f"sigma must be scalar, rank-1 or rank-2. Got rank={sigma.shape.rank}."
            )
        return sigma

    def _conditional_price_batch(
        self,
        spot,
        past_log_sum,
        future_fixing_steps,
        rate_vec,
        sigma_vec,
    ):
        spot = tf.reshape(tf.convert_to_tensor(spot, dtype=tf.float32), (-1,))
        past_log_sum = tf.reshape(tf.convert_to_tensor(past_log_sum, dtype=tf.float32), (-1,))
        rate_vec = tf.reshape(tf.convert_to_tensor(rate_vec, dtype=tf.float32), (-1,))
        sigma_vec = tf.reshape(tf.convert_to_tensor(sigma_vec, dtype=tf.float32), (-1,))

        if not (spot.shape == past_log_sum.shape == rate_vec.shape == sigma_vec.shape):
            raise ValueError(
                "spot, past_log_sum, rate_vec and sigma_vec must share the same shape."
            )

        strike_t = tf.constant(float(self.strike), dtype=tf.float32)
        total_fixings_f = tf.constant(float(self.total_fixings), dtype=tf.float32)
        option_type = str(self.option_type).lower()
        n_future = int(len(future_fixing_steps))

        if n_future == 0:
            deterministic_geo = tf.exp(past_log_sum / total_fixings_f)
            if option_type == "call":
                return tf.maximum(deterministic_geo - strike_t, 0.0)
            if option_type == "put":
                return tf.maximum(strike_t - deterministic_geo, 0.0)
            raise ValueError("option_type must be 'call' or 'put'.")

        tau = np.asarray(future_fixing_steps, dtype=np.float64) * float(self.dt)
        sum_tau_over_total = tf.constant(
            float(np.sum(tau)) / float(self.total_fixings), dtype=tf.float32
        )
        n_future_over_total = tf.constant(
            float(n_future) / float(self.total_fixings), dtype=tf.float32
        )
        var_scale = tf.constant(
            float(np.minimum.outer(tau, tau).sum()) / (float(self.total_fixings) ** 2),
            dtype=tf.float32,
        )
        t_rem = tf.constant(float(np.max(tau)), dtype=tf.float32)

        safe_spot = tf.maximum(spot, tf.constant(1e-12, dtype=tf.float32))
        drift = rate_vec - 0.5 * tf.square(sigma_vec)
        mu = n_future_over_total * tf.math.log(safe_spot) + drift * sum_tau_over_total
        var = tf.maximum(
            tf.square(sigma_vec) * var_scale,
            tf.constant(1e-12, dtype=tf.float32),
        )
        std = tf.sqrt(var)

        m = (past_log_sum / total_fixings_f) + mu
        log_k = tf.math.log(strike_t)
        d1 = (m - log_k + var) / std
        d2 = (m - log_k) / std

        inv_sqrt_2 = tf.constant(1.0 / np.sqrt(2.0), dtype=tf.float32)
        nd1 = 0.5 * (1.0 + tf.math.erf(d1 * inv_sqrt_2))
        nd2 = 0.5 * (1.0 + tf.math.erf(d2 * inv_sqrt_2))
        nmd1 = 1.0 - nd1
        nmd2 = 1.0 - nd2

        exp_term = tf.exp(m + 0.5 * var)
        call_no_disc = exp_term * nd1 - strike_t * nd2
        put_no_disc = strike_t * nmd2 - exp_term * nmd1

        discount = tf.exp(-rate_vec * t_rem)
        if option_type == "call":
            return discount * call_no_disc
        if option_type == "put":
            return discount * put_no_disc
        raise ValueError("option_type must be 'call' or 'put'.")

    def _infer_n_paths(self, path_s0, path_r, path_sigma):
        if path_s0 is not None:
            s = tf.reshape(tf.convert_to_tensor(path_s0, dtype=tf.float32), (-1,))
            if s.shape[0] is not None:
                return int(s.shape[0])
        for values in (path_r, path_sigma):
            if values is None:
                continue
            v = tf.reshape(tf.convert_to_tensor(values, dtype=tf.float32), (-1,))
            if v.shape[0] is not None:
                return int(v.shape[0])
        return 1

    def process_batch(
        self,
        batch_paths,
        batch_T_minus_t,
        batch_path_r=None,
        batch_path_sigma=None,
    ):
        batch_size = batch_paths.shape[0]
        self.reset_last_delta(batch_size)
        self.reset_running_state(batch_size)
        rate_vec = self._normalize_rate_vector(batch_path_r, batch_size)
        n_steps = int(batch_paths.shape[1]) - 1
        sigma_vec = self._normalize_sigma_vector(
            batch_path_sigma,
            batch_size,
            n_steps=n_steps,
        )

        all_actions = []
        for t in range(batch_paths.shape[1] - 1):  # Hedge up to T-1
            current_paths = batch_paths[:, t, :]
            current_spot = tf.maximum(tf.cast(current_paths[:, 0], tf.float32), 1e-12)

            # State at time t includes current fixing if t is in fixing calendar.
            if self._is_fixing_step(t):
                self._running_log_sum += tf.math.log(current_spot)

            future_fixings = self._future_fixing_steps(t)
            sigma_t = sigma_vec
            if sigma_vec.shape.rank == 2:
                sigma_t = sigma_vec[:, t]
            epsilon = tf.maximum(
                tf.abs(current_spot) * float(self.bump_size),
                tf.constant(1e-6, dtype=tf.float32),
            )
            price_up = self._conditional_price_batch(
                spot=current_spot + epsilon,
                past_log_sum=self._running_log_sum,
                future_fixing_steps=future_fixings,
                rate_vec=rate_vec,
                sigma_vec=sigma_t,
            )
            price_down = self._conditional_price_batch(
                spot=tf.maximum(current_spot - epsilon, 1e-8),
                past_log_sum=self._running_log_sum,
                future_fixing_steps=future_fixings,
                rate_vec=rate_vec,
                sigma_vec=sigma_t,
            )
            target_delta = (price_up - price_down) / (2.0 * epsilon)

            action = self._to_actions(target_delta, current_paths)
            all_actions.append(action)

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_size, 1, all_actions.shape[-1]), dtype=tf.float32)
        all_actions = tf.concat([all_actions, zero_action], axis=1)
        return all_actions

    def get_model_price(self):
        price = self.get_model_price_batch(
            path_s0=tf.constant([float(self.S0)], dtype=tf.float32),
            path_r=tf.constant([float(self.r)], dtype=tf.float32),
            path_sigma=tf.constant([float(self.sigma)], dtype=tf.float32),
        )
        return price[0]

    def get_model_price_batch(self, path_s0=None, path_r=None, path_sigma=None):
        n_paths = self._infer_n_paths(path_s0=path_s0, path_r=path_r, path_sigma=path_sigma)
        if path_s0 is None:
            spot = tf.fill((n_paths,), tf.constant(float(self.S0), dtype=tf.float32))
        else:
            spot = tf.reshape(tf.convert_to_tensor(path_s0, dtype=tf.float32), (-1,))
            n_paths = int(spot.shape[0])

        rate_vec = self._normalize_rate_vector(path_r, n_paths)
        sigma_vec = self._normalize_sigma_vector(path_sigma, n_paths, n_steps=int(self.N))
        if sigma_vec.shape.rank == 2:
            sigma_vec = sigma_vec[:, 0]

        past_log_sum = tf.zeros((n_paths,), dtype=tf.float32)
        if self._is_fixing_step(0):
            past_log_sum += tf.math.log(tf.maximum(spot, 1e-12))

        future_fixings = self._future_fixing_steps(0)
        return self._conditional_price_batch(
            spot=spot,
            past_log_sum=past_log_sum,
            future_fixing_steps=future_fixings,
            rate_vec=rate_vec,
            sigma_vec=sigma_vec,
        )
