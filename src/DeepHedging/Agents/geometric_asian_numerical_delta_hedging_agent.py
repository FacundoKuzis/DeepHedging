import tensorflow as tf
import tensorflow_probability as tfp
from DeepHedging.Agents import DeltaHedgingAgent

class GeometricAsianNumericalDeltaHedgingAgent(DeltaHedgingAgent):
    """
    A delta hedging agent for geometric Asian options that computes delta numerically
    using finite differences by bumping the stock price.
    """
    plot_color = 'firebrick' 
    name = 'geometric_asian_numerical_delta_hedging'
    is_trainable = False
    plot_name = {
        'en': 'Geometric Asian Numerical Delta',
        'es': 'Delta de Opción Asiática Geométrica - Diferencias Finitas'
    }

    def __init__(self, gbm_stock, option_class, bump_size=0.01):
        """
        Initialize the agent.

        Arguments:
        - gbm_stock: An instance containing the stock parameters.
        - option_class: An instance of the option class containing option parameters.
        - bump_size (float): The relative size of the bump to compute finite differences.
                             Default is 0.01 (1%).
        """
        super().__init__(gbm_stock, option_class)
        self.name = f'asian_numerical_{bump_size}_delta_hedging'
        self.plot_name['en'] = f"{self.plot_name['en']} with {bump_size*100}% bump"
        self.plot_name['es'] = f"{self.plot_name['es']} con incremento del {bump_size*100}%"
        self.bump_size = bump_size  # Relative bump size for finite differences

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

    def get_model_price(self, S = None, T_minus_t = None):
        """
        Calculate the price for the geometric Asian option using the provided formula.

        Arguments:
        - S (tf.Tensor): Current stock prices. Shape: (...)
        - T_minus_t (tf.Tensor): Time to maturity. Shape: (...)

        Returns:
        - price (tf.Tensor): Option prices. Shape: (...)
        """
        sigma = self.sigma
        r = self.r
        K = self.strike
        T = T_minus_t

        sqrt_3 = tf.sqrt(3.0)
        sqrt_T = tf.sqrt(T)
        
        if S is None:
            S = self.S0
        if T is None:
            T = self.T
            
        z = self.compute_z(S, T)

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
            price = S * Phi_z1 * exp_term1 - K * Phi_z * exp_term2
        elif self.option_type.lower() == 'put':
            price = K * (1 - Phi_z) * exp_term2 - S * (1 - Phi_z1) * exp_term1
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return price

    def delta(self, S, T_minus_t):
        """
        Calculate the delta of the geometric Asian option numerically using finite differences.

        Arguments:
        - S (tf.Tensor): Current stock prices. Shape: (...)
        - T_minus_t (tf.Tensor): Time to maturity. Shape: (...)

        Returns:
        - delta (tf.Tensor): The computed deltas. Shape: (...)
        """
        # Ensure S and T_minus_t are tensors
        S = tf.convert_to_tensor(S, dtype=tf.float32)
        T_minus_t = tf.convert_to_tensor(T_minus_t, dtype=tf.float32)

        # Compute the bump amount
        epsilon = self.bump_size * S

        # Price at S + epsilon
        S_up = S + epsilon
        price_up = self.get_model_price(S_up, T_minus_t)

        # Price at S - epsilon
        S_down = S - epsilon
        price_down = self.get_model_price(S_down, T_minus_t)

        # Numerical delta approximation
        delta = (price_up - price_down) / (2*epsilon)

        return delta
