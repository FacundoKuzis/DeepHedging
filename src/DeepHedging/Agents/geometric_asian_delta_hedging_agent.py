import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np  # For pi
from DeepHedging.Agents import DeltaHedgingAgent

class GeometricAsianDeltaHedgingAgent(DeltaHedgingAgent):
    """
    A delta hedging agent for geometric Asian options.
    """
    plot_color = 'orange' 
    name = 'asian_delta_hedging'
    is_trainable = False
    plot_name = {
        'en': 'Geometric Asian Delta',
        'es': 'Delta de Opción Asiática Geométrica'
    }

    def __init__(self, gbm_stock, option_class):
        super().__init__(gbm_stock, option_class)

    def d1(self, S, T_minus_t):
        """
        Calculate the d1 component used in the geometric Asian option formula.
        """

        numerator = tf.math.log(S / self.strike) + (T_minus_t / 2) * (self.r + (self.sigma ** 2) / 6)
        denominator = self.sigma * tf.sqrt(T_minus_t / 3)
        return numerator / denominator

    def d2(self, S, T_minus_t):
        """
        Calculate the d2 component used in the geometric Asian option formula.
        """

        numerator = tf.math.log(S / self.strike) + (T_minus_t / 2) * (self.r - (self.sigma ** 2) / 2)
        denominator = self.sigma * tf.sqrt(T_minus_t / 3)
        return numerator / denominator

    def delta(self, S, T_minus_t):
        """
        Calculate the delta of the geometric Asian option.
        """

        d1 = self.d1(S, T_minus_t)
        d2 = self.d2(S, T_minus_t)

        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        Nd1 = normal_dist.cdf(d1)
        N_minus_d1 = normal_dist.cdf(-d1)

        exp_term1 = tf.exp(-((self.r + (self.sigma ** 2) / 6) * (T_minus_t / 2)))
        exp_term2 = tf.exp(-((self.r * (T_minus_t / 2)) + (T_minus_t * (self.sigma ** 2) / 12) + (d1 ** 2) / 2))
        exp_term3 = tf.exp(-((self.r * T_minus_t) + (d2 ** 2) / 2))

        if self.option_type == 'call':
            first_term = exp_term1 * Nd1
            second_term = (1 / (self.sigma * tf.sqrt(2 * np.pi * T_minus_t / 3))) * (
                exp_term2 - (self.strike / S) * exp_term3
            )
            delta = first_term + second_term
        elif self.option_type == 'put':
            first_term = -exp_term1 * N_minus_d1
            second_term = (1 / (self.sigma * tf.sqrt(2 * np.pi * T_minus_t / 3))) * (
                -exp_term2 + (self.strike / S) * exp_term3
            )
            delta = first_term + second_term
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return delta

    def get_model_price(self):
        """
        Calculate the price for the geometric Asian option.
        """
        eps = 1e-4
        T_tilde = self.T + eps  # Total time to maturity

        d1 = self.d1(self.S0, T_tilde)
        d2 = self.d2(self.S0, T_tilde)

        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        Nd1 = normal_dist.cdf(d1)
        Nd2 = normal_dist.cdf(d2)
        N_minus_d1 = normal_dist.cdf(-d1)
        N_minus_d2 = normal_dist.cdf(-d2)

        exp_term1 = tf.exp(-((self.r + (self.sigma ** 2) / 6) * (T_tilde / 2)))

        if self.option_type == 'call':
            price = self.S0 * exp_term1 * Nd1 - self.strike * tf.exp(-self.r * T_tilde) * Nd2
        elif self.option_type == 'put':
            price = self.strike * tf.exp(-self.r * T_tilde) * N_minus_d2 - self.S0 * exp_term1 * N_minus_d1
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return price
