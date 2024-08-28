import tensorflow as tf
import tensorflow_probability as tfp
from DeepHedging.Agents import BaseAgent

class DeltaHedgingAgent(BaseAgent):
    """
    A delta hedging agent that computes the delta of an option and uses it as the hedging strategy.

    Arguments:
    - gbm_stock (GBMStock): An instance of the GBMStock class containing the stock parameters.
    - strike (float): Strike price of the option.
    - option_type (str): Type of the option ('call' or 'put').
    """

    def __init__(self, gbm_stock, option_class):
        self.S0 = gbm_stock.S0
        self.T = gbm_stock.T
        self.N = gbm_stock.N
        self.r = gbm_stock.r
        self.sigma = gbm_stock.sigma
        self.strike = option_class.strike
        self.option_type = option_class.option_type
        self.dt = gbm_stock.dt
        self.name = 'delta_hedging'

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
        return (tf.math.log(S / self.strike) + (self.r + 0.5 * self.sigma ** 2) * (T_minus_t + eps)) / (self.sigma * tf.sqrt(T_minus_t + eps))

    def delta(self, S, T_minus_t):
        """
        Calculate the delta of the option.

        Arguments:
        - S (tf.Tensor): The current stock price.
        - t (tf.Tensor): The current time.

        Returns:
        - delta (tf.Tensor): The delta value.
        """
        d1 = self.d1(S, T_minus_t)
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        if self.option_type == 'call':
            return normal_dist.cdf(d1)
        elif self.option_type == 'put':
            return normal_dist.cdf(d1) - 1.0
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the delta hedging strategy.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The delta value used as the hedging action.
        """

        delta = self.delta(instrument_paths, T_minus_t)
        action = delta - self.last_delta
        self.last_delta = delta

        return tf.expand_dims(action, axis=-1)

    def reset_last_delta(self, batch_size):
        self.last_delta = tf.zeros((batch_size,), dtype=tf.float32)

    def process_batch(self, batch_paths, batch_T_minus_t):
        self.reset_last_delta(batch_paths.shape[0])
        all_actions = []
        for t in range(batch_paths.shape[1] -1):  # timesteps until T-1
            current_paths = batch_paths[:, t]
            current_T_minus_t = batch_T_minus_t[:, t]
            action = self.act(current_paths, current_T_minus_t)
            all_actions.append(action)

        all_actions = tf.concat(all_actions, axis=1)
        zero_action = tf.zeros((batch_paths.shape[0], 1))
        all_actions = tf.concat([all_actions, zero_action], axis=1)

        return all_actions