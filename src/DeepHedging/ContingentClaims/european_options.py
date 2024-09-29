import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from DeepHedging.ContingentClaims import ContingentClaim

class EuropeanCall(ContingentClaim):
    """
    A class representing a European call option.

    Arguments:
    - strike (float): The strike price of the option.
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the European call option.
    """

    def __init__(self, strike, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.option_type = 'call'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the European call option.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the European call option.
                              Shape is (num_paths, N+1).

        """
        # The payoff is max(S(T) - K, 0) for a European call option
        payoff = tf.maximum(paths[:, -1] - self.strike, 0) * self.amount
        return payoff

class EuropeanPut(ContingentClaim):
    """
    A class representing a European put option.

    Arguments:
    - strike (float): The strike price of the option.
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the European put option.
    """

    def __init__(self, strike, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.option_type = 'put'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the European put option.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the European put option.
                               Shape is (num_paths, N+1).

        """
        # The payoff is max(K - S(T), 0) for a European put option
        payoff = tf.maximum(self.strike - paths[:, -1], 0) * self.amount
        return payoff


class ChooserOption(ContingentClaim):
    """
    A class representing a chooser option.

    Arguments:
    - strike (float): The strike price of the option.
    - t_choice (float): Time until the choice date from inception.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.
    - T (float): Time to maturity.
    - N (int): Number of time steps in the simulation.
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the chooser option.
    """

    def __init__(self, strike, t_choice, r, sigma, T, N, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.t_choice = t_choice
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T / N
        self.times = np.linspace(0, T, N + 1)  # Shape: (N+1,)
        self.option_type = 'call'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the chooser option.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the chooser option.
                              Shape is (num_paths,).
        """
        eps = 1e-6

        # Find the index corresponding to t_choice
        t_choice_index = np.searchsorted(self.times, self.t_choice, side='right') - 1

        # Get the stock price at t_choice and at maturity T
        S_t_choice = paths[:, t_choice_index]  # Shape: (num_paths,)
        S_T = paths[:, -1]  # Shape: (num_paths,)

        # Time to maturity from t_choice
        T_minus_t_choice = self.T - self.t_choice
        T_minus_t_choice = max(T_minus_t_choice, eps)

        # Compute d1 and d2 for call and put options at t_choice
        ln_S_over_X = tf.math.log(S_t_choice / self.strike)
        denominator = self.sigma * np.sqrt(T_minus_t_choice)
        d1 = (ln_S_over_X + (self.r + 0.5 * self.sigma ** 2) * T_minus_t_choice) / denominator
        d2 = d1 - denominator

        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        N_d1 = normal_dist.cdf(d1)
        N_minus_d1 = normal_dist.cdf(-d1)
        N_d2 = normal_dist.cdf(d2)
        N_minus_d2 = normal_dist.cdf(-d2)

        # Compute option prices at t_choice
        call_prices = (S_t_choice * N_d1 - self.strike * tf.exp(-self.r * T_minus_t_choice) * N_d2)
        put_prices = (self.strike * tf.exp(-self.r * T_minus_t_choice) * N_minus_d2 - S_t_choice * N_minus_d1)

        # Decide which option is more valuable
        is_call_more_valuable = call_prices > put_prices  # Shape: (num_paths,)

        # Compute payoffs at T
        call_payoff = tf.maximum(S_T - self.strike, 0.0)
        put_payoff = tf.maximum(self.strike - S_T, 0.0)

        # Select the payoff based on the chosen option
        payoff = tf.where(is_call_more_valuable, call_payoff, put_payoff) * self.amount

        return payoff
