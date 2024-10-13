import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np  # For pi
from DeepHedging.Agents import DeltaHedgingAgent

class AsianDeltaHedgingAgent2(DeltaHedgingAgent):
    """
    A delta hedging agent for geometric Asian options, using the provided option pricing formula.
    """

    def __init__(self, gbm_stock, option_class):
        super().__init__(gbm_stock, option_class)
        self.name = 'asian_delta_hedging_custom_formula'
        self.plot_name = 'Geometric Asian Delta (Custom Formula)'

    def compute_z(self, S, T_minus_t):
        """
        Compute z as per the provided formula.

        z = [ sqrt(3) * (4 * ln(S/K) + T * (2r - sigma^2)) ] / [4 * sqrt(T) * sigma]
        """
        sigma = self.sigma
        r = self.r
        K = self.strike
        T = T_minus_t  # Time to maturity

        sqrt_3 = tf.sqrt(3.0)
        ln_S_over_K = tf.math.log(S / K)
        numerator = sqrt_3 * (4 * ln_S_over_K + T * (2 * r - sigma ** 2))
        denominator = 4 * tf.sqrt(T) * sigma

        z = numerator / denominator
        return z

    def get_model_price(self):
        """
        Calculate the price for the geometric Asian option using the provided formula.
        """
        S0 = self.S0
        sigma = self.sigma
        r = self.r
        K = self.strike
        T = self.T

        sqrt_3 = tf.sqrt(3.0)
        sqrt_T = tf.sqrt(T)

        z = self.compute_z(S0, T)

        # Compute z1 = z + (sigma * sqrt(T)) / sqrt(3)
        z1 = z + (sigma * sqrt_T) / sqrt_3

        # Compute the exponential terms
        exp_term1 = tf.exp(- (6 * r * T + sigma ** 2 * T) / 12)
        exp_term2 = tf.exp(- r * T)

        # Compute the cumulative distribution functions
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        Phi_z1 = normal_dist.cdf(z1)
        Phi_z = normal_dist.cdf(z)

        # Compute the option price
        if self.option_type.lower() == 'call':
            price = S0 * Phi_z1 * exp_term1 - K * Phi_z * exp_term2
        elif self.option_type.lower() == 'put':
            price = K * (1 - Phi_z) * exp_term2 - S0 * (1 - Phi_z1) * exp_term1
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return price

    def delta(self, S, T_minus_t):
        """
        Calculate the delta of the geometric Asian option using the derivative of the provided formula.
        """
        sigma = self.sigma
        r = self.r
        K = self.strike
        T = T_minus_t

        sqrt_3 = tf.sqrt(3.0)
        sqrt_T = tf.sqrt(T)

        z = self.compute_z(S, T)
        z1 = z + (sigma * sqrt_T) / sqrt_3

        # Compute the exponential terms
        exp_term1 = tf.exp(- (6 * r * T + sigma ** 2 * T) / 12)
        exp_term2 = tf.exp(- r * T)

        # Compute A and B
        A = S * exp_term1
        B = K * exp_term2

        # Compute the cumulative distribution functions and PDFs
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        Phi_z1 = normal_dist.cdf(z1)
        Phi_z = normal_dist.cdf(z)
        phi_z1 = normal_dist.prob(z1)  # PDF at z1
        phi_z = normal_dist.prob(z)    # PDF at z

        # Compute dz/dS
        dz_dS = sqrt_3 / (S * sqrt_T * sigma)

        # Compute delta
        if self.option_type.lower() == 'call':
            # For call options
            delta = exp_term1 * Phi_z1 + dz_dS * (A * phi_z1 - B * phi_z)
        elif self.option_type.lower() == 'put':
            # For put options
            delta = -exp_term1 * (1 - Phi_z1) + dz_dS * (-A * phi_z1 + B * phi_z)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return delta
