import tensorflow as tf
import tensorflow_probability as tfp
from DeepHedging.Agents import BaseAgent
import logging


logger = logging.getLogger(__name__)

class DeltaHedgingAgent(BaseAgent):
    """
    A delta hedging agent that computes the delta of an option and uses it as the hedging strategy.

    Arguments:
    - stock_model (GBMStock): An instance of the GBMStock class containing the stock parameters.
    - strike (float): Strike price of the option.
    - option_type (str): Type of the option ('call' or 'put').
    """

    plot_color = 'orange' 
    name = 'bs_delta_hedging'
    is_trainable = False
    plot_name = {
        'en': 'BS European Vanilla Delta',
        'es': 'Agente delta de opción europea'
    }

    def __init__(
        self,
        stock_model,
        option_class,
        no_trade_band=0.0,
        no_trade_band_mode="absolute",
        no_trade_band_eps=1e-8,
    ):
        self.stock_model = stock_model
        self.S0 = stock_model.S0
        self.T = stock_model.T
        self.N = stock_model.N
        self.r = stock_model.r
        self.sigma = stock_model.sigma
        self.strike = option_class.strike
        self.option_type = option_class.option_type
        self.dt = stock_model.dt
        self.no_trade_band = float(no_trade_band)
        self.no_trade_band_eps = float(no_trade_band_eps)
        self.set_no_trade_band_mode(no_trade_band_mode)

    def build_model(self):
        """
        Dummy implementation as no model building is required for delta hedging.
        """
        pass

    def d1(self, S, T_minus_t):
        """
        Calculate the d1 component used in the Black-Scholes formula.

        Arguments:
        - S (tf.Tensor): The current stock price.
        - T_minus_t (tf.Tensor): The current T - t.

        Returns:
        - d1 (tf.Tensor): The d1 value.
        """

        eps = 1e-4
        return self._d1_with_rate(S=S, T_minus_t=T_minus_t, rate=self.r, sigma=self.sigma)

    def _normalize_rate_input(self, rate, batch_size):
        return self._normalize_vector_input(rate, batch_size, field_name="rate")

    def _normalize_sigma_input(self, sigma, batch_size, n_steps=None):
        sigma_tensor = tf.convert_to_tensor(sigma, dtype=tf.float32)
        if sigma_tensor.shape.rank == 0:
            return tf.maximum(sigma_tensor, tf.constant(1e-8, dtype=tf.float32))

        if sigma_tensor.shape.rank == 1:
            sigma_tensor = tf.reshape(sigma_tensor, (-1,))
            expected = None if batch_size is None else int(batch_size)
            observed = None if sigma_tensor.shape[0] is None else int(sigma_tensor.shape[0])
            if expected is not None and observed is not None and observed != expected:
                raise ValueError(
                    f"sigma length mismatch: expected {expected}, got {observed}."
                )
            return tf.maximum(sigma_tensor, tf.constant(1e-8, dtype=tf.float32))

        if sigma_tensor.shape.rank == 2:
            expected = None if batch_size is None else int(batch_size)
            observed = None if sigma_tensor.shape[0] is None else int(sigma_tensor.shape[0])
            if expected is not None and observed is not None and observed != expected:
                raise ValueError(
                    f"sigma batch mismatch: expected {expected}, got {observed}."
                )
            if n_steps is not None:
                observed_steps = None if sigma_tensor.shape[1] is None else int(sigma_tensor.shape[1])
                if observed_steps is not None and observed_steps < int(n_steps):
                    raise ValueError(
                        f"sigma timestep mismatch: expected at least {int(n_steps)}, got {observed_steps}."
                    )
            return tf.maximum(sigma_tensor, tf.constant(1e-8, dtype=tf.float32))

        raise ValueError(
            f"sigma must be scalar, rank-1 or rank-2 tensor. Got rank={sigma_tensor.shape.rank}."
        )

    def _normalize_vector_input(self, values, batch_size, field_name):
        values_tensor = tf.convert_to_tensor(values, dtype=tf.float32)
        if values_tensor.shape.rank == 0:
            return values_tensor
        values_tensor = tf.reshape(values_tensor, (-1,))
        expected = None if batch_size is None else int(batch_size)
        observed = None if values_tensor.shape[0] is None else int(values_tensor.shape[0])
        if expected is not None and observed is not None and observed != expected:
            raise ValueError(
                f"{field_name} length mismatch: expected {expected}, got {observed}."
            )
        return values_tensor

    def _d1_with_rate(self, S, T_minus_t, rate, sigma):
        S = tf.convert_to_tensor(S, dtype=tf.float32)
        T_minus_t = tf.convert_to_tensor(T_minus_t, dtype=tf.float32)
        r_eff = self._normalize_rate_input(rate, tf.shape(S)[0] if S.shape.rank > 0 else None)
        sigma_eff = self._normalize_sigma_input(sigma, tf.shape(S)[0] if S.shape.rank > 0 else None)
        eps = tf.constant(1e-4, dtype=tf.float32)
        return (
            tf.math.log(S / self.strike)
            + (r_eff + 0.5 * tf.square(sigma_eff)) * (T_minus_t + eps)
        ) / (sigma_eff * tf.sqrt(T_minus_t + eps))

    def delta(self, S, T_minus_t, rate=None, sigma=None):
        """
        Calculate the delta of the option.

        Arguments:
        - S (tf.Tensor): The current stock price.
        - t (tf.Tensor): The current time.

        Returns:
        - delta (tf.Tensor): The delta value.
        """
        sigma_eff = self.sigma if sigma is None else sigma
        if rate is None:
            d1 = self._d1_with_rate(S=S, T_minus_t=T_minus_t, rate=self.r, sigma=sigma_eff)
        else:
            d1 = self._d1_with_rate(S=S, T_minus_t=T_minus_t, rate=rate, sigma=sigma_eff)
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        if self.option_type == 'call':
            return normal_dist.cdf(d1)
        elif self.option_type == 'put':
            return normal_dist.cdf(d1) - 1.0
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

    def act(self, instrument_paths, T_minus_t, rate=None, sigma=None):
        """
        Act based on the delta hedging strategy.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The delta value used as the hedging action.
        """

        target_delta = self.delta(
            instrument_paths[:, 0],
            T_minus_t,
            rate=rate,
            sigma=sigma,
        ) # ASSUMPTION: Stock is the first instrument
        return self._to_actions(target_delta, instrument_paths)

    def set_no_trade_band(self, no_trade_band):
        self.no_trade_band = float(no_trade_band)

    def set_no_trade_band_mode(self, no_trade_band_mode):
        mode = str(no_trade_band_mode).strip().lower()
        if mode not in {"absolute", "percentage"}:
            raise ValueError("no_trade_band_mode must be 'absolute' or 'percentage'.")
        self.no_trade_band_mode = mode

    def _apply_no_trade_band(self, target_delta):
        if self.no_trade_band <= 0.0:
            return target_delta
        diff = tf.abs(target_delta - self.last_delta)
        if self.no_trade_band_mode == "percentage":
            scale = tf.maximum(tf.abs(target_delta), tf.abs(self.last_delta))
            scale = tf.maximum(scale, tf.constant(self.no_trade_band_eps, dtype=tf.float32))
            relative_diff = diff / scale
            return tf.where(relative_diff < self.no_trade_band, self.last_delta, target_delta)
        return tf.where(diff < self.no_trade_band, self.last_delta, target_delta)

    def _to_actions(self, target_delta, instrument_paths):
        effective_delta = self._apply_no_trade_band(tf.cast(target_delta, tf.float32))
        action = effective_delta - self.last_delta
        self.last_delta = effective_delta

        action = tf.expand_dims(action, axis=-1)
        zeros = tf.zeros((instrument_paths.shape[0], instrument_paths.shape[1] - 1), dtype=tf.float32)
        actions = tf.concat([action, zeros], axis=1)
        return actions

    def reset_last_delta(self, batch_size):
        self.last_delta = tf.zeros((batch_size,), dtype=tf.float32)

    def process_batch(self, batch_paths, batch_T_minus_t, batch_path_r=None, batch_path_sigma=None):
        if batch_path_r is not None:
            rate_vector = self._normalize_rate_input(batch_path_r, batch_paths.shape[0])
        else:
            rate_vector = None
        n_steps = int(batch_paths.shape[1]) - 1
        if batch_path_sigma is not None:
            sigma_vector = self._normalize_sigma_input(
                batch_path_sigma,
                batch_paths.shape[0],
                n_steps=n_steps,
            )
        else:
            sigma_vector = None
        self.reset_last_delta(batch_paths.shape[0])
        all_actions = []
        for t in range(batch_paths.shape[1] -1):  # timesteps until T-1
            logger.debug("Processing delta hedging timestep %s", t)
            current_paths = batch_paths[:, t, :] # (n_simulations, n_timesteps, n_instruments)
            current_T_minus_t = batch_T_minus_t[:, t] # (n_simulations, n_timesteps)
            sigma_t = sigma_vector
            if sigma_vector is not None and sigma_vector.shape.rank == 2:
                sigma_t = sigma_vector[:, t]
            action = self.act(
                current_paths,
                current_T_minus_t,
                rate=rate_vector,
                sigma=sigma_t,
            )
            all_actions.append(action)

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[-1]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)

        return all_actions # (n_simulations, n_timesteps, n_instruments)

    def get_model_price(self):
        """
        Calculate the Black-Scholes price for the option.

        Returns:
        - price (tf.Tensor): The Black-Scholes price of the option.
        """
        price = self.get_model_price_batch(
            path_s0=tf.constant([float(self.S0)], dtype=tf.float32),
            path_r=tf.constant([float(self.r)], dtype=tf.float32),
            path_sigma=tf.constant([float(self.sigma)], dtype=tf.float32),
        )
        return price[0]

    def get_model_price_batch(self, path_s0=None, path_r=None, path_sigma=None):
        """
        Vectorized Black-Scholes pricing for path-specific inputs.

        Arguments:
        - path_s0 (tensor/array/scalar|None): initial spot per path; defaults to self.S0.
        - path_r (tensor/array/scalar|None): risk-free rate per path; defaults to self.r.
        - path_sigma (tensor/array/scalar|None): volatility per path; defaults to self.sigma.

        Returns:
        - prices (tf.Tensor): shape (n_paths,)
        """
        if path_s0 is None:
            S = tf.constant([float(self.S0)], dtype=tf.float32)
        else:
            S = tf.reshape(tf.convert_to_tensor(path_s0, dtype=tf.float32), (-1,))

        n_paths = int(S.shape[0])
        r_eff = self._normalize_rate_input(self.r if path_r is None else path_r, n_paths)
        sigma_eff = tf.convert_to_tensor(
            float(self.sigma) if path_sigma is None else path_sigma,
            dtype=tf.float32,
        )
        if sigma_eff.shape.rank == 0:
            sigma_eff = tf.fill((n_paths,), sigma_eff)
        else:
            if sigma_eff.shape.rank == 2:
                sigma_eff = sigma_eff[:, 0]
            sigma_eff = tf.reshape(sigma_eff, (-1,))
            if sigma_eff.shape[0] is not None and int(sigma_eff.shape[0]) != n_paths:
                raise ValueError(
                    f"sigma length mismatch: expected {n_paths}, got {int(sigma_eff.shape[0])}."
                )

        T = tf.constant(float(self.T), dtype=tf.float32)
        strike = tf.constant(float(self.strike), dtype=tf.float32)
        eps = tf.constant(1e-8, dtype=tf.float32)
        sqrt_T = tf.sqrt(T + eps)
        sigma_safe = tf.maximum(sigma_eff, eps)
        d1 = (
            tf.math.log(tf.maximum(S, eps) / strike)
            + (r_eff + 0.5 * tf.square(sigma_safe)) * (T + eps)
        ) / (sigma_safe * sqrt_T)
        d2 = d1 - sigma_safe * sqrt_T

        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        disc = tf.exp(-r_eff * T)
        if self.option_type == 'call':
            return S * normal_dist.cdf(d1) - strike * disc * normal_dist.cdf(d2)
        if self.option_type == 'put':
            return strike * disc * normal_dist.cdf(-d2) - S * normal_dist.cdf(-d1)
        raise ValueError("Option type must be either 'call' or 'put'.")
