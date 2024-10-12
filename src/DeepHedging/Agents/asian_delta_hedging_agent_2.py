import tensorflow as tf
import tensorflow_probability as tfp
from DeepHedging.Agents import DeltaHedgingAgent

class AsianDeltaHedgingAgent2(DeltaHedgingAgent):
    """
    A delta hedging agent for geometric Asian options using the new delta and price calculation without dividend yield.
    """

    def __init__(self, gbm_stock, option_class):
        super().__init__(gbm_stock, option_class)
        self.name = 'asian_delta_hedging'
        self.plot_name = 'Geometric Asian Delta'

    def delta(self, S, T_minus_t):
        """
        Calculate the delta of the geometric Asian option.
        """

        # Parameters
        N = self.N
        sigma = self.sigma
        r = self.r
        K = self.strike
        T = T_minus_t  # Remaining time to maturity

        # Compute mu
        mu = r + 0.5 * sigma ** 2

        # Compute a
        a = N * (N + 1) * (2 * N + 1) / 6

        # Compute sigma_avg
        sigma_avg = sigma * tf.sqrt((2 * N + 1) / (6 * (N + 1)))

        # Compute V
        exponent = ((N + 1) * mu * T) / 2 + (a * sigma ** 2 * T) / (2 * N ** 3)
        V = tf.exp(-r * T) * S * tf.exp(exponent)

        # Compute D1
        D1 = (tf.math.log(V / K) + (r + 0.5 * sigma_avg ** 2) * T) / (sigma_avg * tf.sqrt(T))

        # Compute delta
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        Nd1 = normal_dist.cdf(D1)

        if self.option_type == 'call':
            delta = V * Nd1
        elif self.option_type == 'put':
            delta = V * (Nd1 - 1)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return delta

    def get_model_price(self):
        """
        Calculate the price for the geometric Asian option.
        """
        # Parameters
        N = self.N
        sigma = self.sigma
        r = self.r
        K = self.strike
        T = self.T
        S = self.S0

        # Compute mu
        mu = r + 0.5 * sigma ** 2

        # Compute a
        a = N * (N + 1) * (2 * N + 1) / 6

        # Compute sigma_avg
        sigma_avg = sigma * tf.sqrt((2 * N + 1) / (6 * (N + 1)))

        # Compute V
        exponent = ((N + 1) * mu * T) / 2 + (a * sigma ** 2 * T) / (2 * N ** 3)
        V = tf.exp(-r * T) * S * tf.exp(exponent)

        # Compute D1 and D2
        D1 = (tf.math.log(V / K) + (r + 0.5 * sigma_avg ** 2) * T) / (sigma_avg * tf.sqrt(T))
        D2 = D1 - sigma_avg * tf.sqrt(T)

        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        Nd1 = normal_dist.cdf(D1)
        Nd2 = normal_dist.cdf(D2)
        N_minus_d1 = normal_dist.cdf(-D1)
        N_minus_d2 = normal_dist.cdf(-D2)

        if self.option_type == 'call':
            price = V * Nd1 - tf.exp(-r * T) * K * Nd2
        elif self.option_type == 'put':
            price = tf.exp(-r * T) * K * N_minus_d2 - V * N_minus_d1
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return price
