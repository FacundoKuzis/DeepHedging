import tensorflow as tf
import numpy as np
from DeepHedging.HedgingInstruments import GBMStock, HestonStock
from DeepHedging.ContingentClaims import ContingentClaim

class MonteCarloPricer:
    """
    A Monte Carlo pricer for options using TensorFlow.

    This class simulates asset price paths using a provided Stock model,
    computes option payoffs using a provided ContingentClaim class, and estimates
    option prices and Greeks (e.g., Delta) using finite difference methods.

    Attributes:
        stock_model (Stock): An instance of a subclass of Stock for path simulation.
        r (tf.Tensor): Risk-free interest rate as a float32 tensor.
        T (tf.Tensor): Time to maturity in years as a float32 tensor.
        num_simulations (int): Number of Monte Carlo simulations.
        seed (int, optional): Random seed for reproducibility.
    """

    def __init__(self, stock_model, r, T, num_simulations=10_000, seed=None):
        """
        Initializes the Monte Carlo pricer with a stock model and market parameters.

        Args:
            stock_model (Stock): An instance of a subclass of Stock for path simulation.
            r (float): Risk-free interest rate.
            T (float): Time to maturity in years.
            num_simulations (int, optional): Number of Monte Carlo simulations. Default is 10,000.
            seed (int, optional): Random seed for reproducibility. Default is None.
        """
        self.stock_model = stock_model  # Instance of Stock subclass (e.g., GBMStock, HestonStock)
        self.r = tf.constant(r, dtype=tf.float32)
        self.T = tf.constant(T, dtype=tf.float32)
        self.num_simulations = num_simulations
        self.seed = seed

        # Set random seed for reproducibility if provided
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

    def simulate_paths(self):
        """
        Simulates asset price paths using the provided stock model.

        Returns:
            tf.Tensor: Simulated asset paths of shape (num_simulations, N+1).
                       For models returning multiple paths (e.g., Heston), only the stock paths are returned.
        """
        if isinstance(self.stock_model, HestonStock):
            # If the stock model returns variance paths, discard them
            S_paths, _ = self.stock_model.generate_paths(num_paths=self.num_simulations, random_seed=self.seed)
            return S_paths  # Shape: (num_simulations, N+1)
        else:
            # For models that return only stock paths
            S_paths = self.stock_model.generate_paths(num_paths=self.num_simulations, random_seed=self.seed)
            return S_paths  # Shape: (num_simulations, N+1)

    def price(self, contingent_claim, paths = None):
        """
        Prices the option using Monte Carlo simulation.

        Args:
            contingent_claim (ContingentClaim): An instance of a class inheriting from ContingentClaim,
                                               which defines the option payoff.

        Returns:
            float: Estimated present value of the option.
        """
        if paths is None:
            # Simulate asset paths
            paths = self.simulate_paths()  # Shape: (num_simulations, N+1)

        # Calculate payoffs using the contingent claim's calculate_payoff method
        payoffs = contingent_claim.calculate_payoff(paths)  # Shape: (num_simulations,)

        # Ensure payoffs are float32
        payoffs = tf.cast(payoffs, tf.float32)

        # Average the payoffs and discount to present value
        discounted_payoff = tf.exp(-self.r * self.T) * tf.reduce_mean(payoffs)

        return discounted_payoff.numpy()

    def delta(self, contingent_claim, bump_size=0.01):
        """
        Estimates the Delta of the option using central finite differences.

        Args:
            contingent_claim (ContingentClaim): An instance of a class inheriting from ContingentClaim,
                                               which defines the option payoff.
            bump_size (float, optional): Relative bump size for finite differences. Default is 1%.

        Returns:
            float: Estimated Delta of the option.
        """
        # Calculate epsilon based on bump_size
        epsilon = self.stock_model.S0 * bump_size

        # Price at S0 + epsilon
        original_S0 = self.stock_model.S0  # Store original S0
        self.stock_model.S0 += epsilon
        price_up = self.price(contingent_claim)

        # Price at S0 - epsilon
        self.stock_model.S0 = original_S0 - epsilon
        price_down = self.price(contingent_claim)

        # Restore original S0
        self.stock_model.S0 = original_S0

        # Central difference approximation of Delta
        delta = (price_up - price_down) / (2 * epsilon)

        return delta

    def price_with_S0(self, contingent_claim, S0, paths=None):
        """
        Prices the option using Monte Carlo simulation with a specified initial asset price.

        Args:
            contingent_claim (ContingentClaim): An instance of a class inheriting from ContingentClaim,
                                               which defines the option payoff.
            S0 (float): Initial asset price for this pricing.
            paths (tf.Tensor, optional): Pre-simulated paths with the new S0. If provided, these paths are used.

        Returns:
            float: Estimated present value of the option with the given S0.
        """
        # Temporarily set the stock model's S0 to the new value
        original_S0 = self.stock_model.S0
        self.stock_model.S0 = S0

        # Simulate asset paths
        if paths is None:
            paths = self.simulate_paths()

        # Calculate payoffs using the contingent claim's calculate_payoff method
        payoffs = contingent_claim.calculate_payoff(paths)  # Shape: (num_simulations,)

        # Ensure payoffs are float32
        payoffs = tf.cast(payoffs, tf.float32)

        # Average the payoffs and discount to present value
        discounted_payoff = tf.exp(-self.r * self.T) * tf.reduce_mean(payoffs)

        # Restore the original S0
        self.stock_model.S0 = original_S0

        return discounted_payoff.numpy()
