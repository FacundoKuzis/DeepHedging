import tensorflow as tf
import numpy as np
from DeepHedging.Agents import BaseAgent, DeltaHedgingAgent, GeometricAsianDeltaHedgingAgent
from DeepHedging.HedgingInstruments import GBMStock, HestonStock
from DeepHedging.ContingentClaims import (
    ContingentClaim,
    AsianArithmeticCall,
    AsianArithmeticPut,
    AsianGeometricCall,
    AsianGeometricPut,
)
from DeepHedging.utils import MonteCarloPricer

class ArithmeticAsianControlVariateAgent(DeltaHedgingAgent):
    """
    An agent that prices and delta hedges an arithmetic Asian option using Monte Carlo simulation with control variate.
    """

    plot_color = 'purple'
    name = 'arithmetic_asian_control_variate'
    is_trainable = False
    plot_name = {
        'en': 'Arithmetic Asian Delta with Control Variate',
        'es': 'Delta de Asiática Aritmética con Variable de Control'
    }

    def __init__(self, stock_model, option_class, num_simulations=10000, bump_size=0.01, seed=33):
        """
        Initialize the agent.

        Args:
            stock_model (Stock): An instance of a Stock subclass (e.g., GBMStock).
            option_class (AsianArithmeticCall or AsianArithmeticPut): An instance of the arithmetic Asian option class.
            num_simulations (int): Number of Monte Carlo simulations.
            bump_size (float): Relative size of the bump for finite differences.
            seed (int): Random seed for reproducibility.
        """
        self.stock_model = stock_model
        self.option_class = option_class  # The arithmetic Asian option
        self.num_simulations = num_simulations
        self.bump_size = bump_size
        self.seed = seed

        self.S0 = stock_model.S0
        self.T = stock_model.T
        self.N = stock_model.N
        self.r = stock_model.r
        self.sigma = stock_model.sigma
        self.strike = option_class.strike
        self.option_type = option_class.option_type
        self.dt = stock_model.dt

        # Initialize last delta for delta hedging
        self.last_delta = None

        # Instantiate Monte Carlo Pricer
        self.pricer = MonteCarloPricer(
            stock_model=self.stock_model,
            r=self.r,
            T=self.T,
            num_simulations=self.num_simulations,
            seed=self.seed
        )

        # Instantiate GeometricAsianDeltaHedgingAgent to get analytical price and delta
        if self.option_type == 'call':
            self.geometric_option = AsianGeometricCall(strike=self.strike)
        elif self.option_type == 'put':
            self.geometric_option = AsianGeometricPut(strike=self.strike)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        self.geometric_agent = GeometricAsianDeltaHedgingAgent(stock_model=self.stock_model, option_class=self.geometric_option)

    def build_model(self):
        """
        No neural network model is needed for this agent.
        """
        pass

    def price(self):
        """
        Price the arithmetic Asian option using Monte Carlo simulation with control variate.

        Returns:
            float: Estimated present value of the arithmetic Asian option using control variate.
        """
        # Simulate paths
        paths = self.pricer.simulate_paths()

        # Calculate payoffs of the arithmetic and geometric Asian options
        arithmetic_payoffs = self.option_class.calculate_payoff(paths)
        geometric_payoffs = self.geometric_option.calculate_payoff(paths)

        # Monte Carlo estimates
        arithmetic_mc_price = tf.exp(-self.r * self.T) * tf.reduce_mean(arithmetic_payoffs)
        geometric_mc_price = tf.exp(-self.r * self.T) * tf.reduce_mean(geometric_payoffs)

        # Analytical price of geometric Asian option
        geometric_analytic_price = self.geometric_agent.get_model_price()

        # Adjusted price using control variate
        adjusted_price = arithmetic_mc_price + (geometric_analytic_price - geometric_mc_price)

        return adjusted_price.numpy()

    def delta(self, bump_size=0.01):
        """
        Estimate the Delta of the option using central finite differences with control variate adjustment.

        Args:
            bump_size (float, optional): Relative bump size for finite differences. Default is 1%.

        Returns:
            float: Estimated Delta of the option.
        """
        epsilon = self.stock_model.S0 * bump_size

        # Price at S0 + epsilon
        original_S0 = self.stock_model.S0
        self.stock_model.S0 = original_S0 + epsilon

        # Update pricer and agents with bumped S0
        self.pricer.stock_model.S0 = self.stock_model.S0
        self.geometric_agent.stock_model.S0 = self.stock_model.S0
        price_up = self.price()

        # Price at S0 - epsilon
        self.stock_model.S0 = original_S0 - epsilon

        # Update pricer and agents with bumped S0
        self.pricer.stock_model.S0 = self.stock_model.S0
        self.geometric_agent.stock_model.S0 = self.stock_model.S0
        price_down = self.price()

        # Restore original S0
        self.stock_model.S0 = original_S0
        self.pricer.stock_model.S0 = original_S0
        self.geometric_agent.stock_model.S0 = original_S0

        # Central difference approximation of Delta
        delta = (price_up - price_down) / (2 * epsilon)

        return delta

    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the Monte Carlo delta hedging strategy with control variate.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
                                        Shape: (batch_size, n_instruments)
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.
                                 Shape: (batch_size,)

        Returns:
        - actions (tf.Tensor): The delta hedging actions for each instrument.
                               Shape: (batch_size, n_instruments)
        """
        # Ensure last_delta is initialized
        if self.last_delta is None:
            self.reset_last_delta(instrument_paths.shape[0])

        # Compute delta
        delta = tf.numpy_function(
            self.compute_deltas,
            [instrument_paths[:, 0].numpy(), T_minus_t.numpy()],
            tf.float32
        )

        # Set shape information
        delta.set_shape((instrument_paths.shape[0],))

        # Calculate action as change in delta
        action = delta - self.last_delta
        self.last_delta = delta

        # Expand dimensions to match the number of instruments
        action = tf.expand_dims(action, axis=-1)  # Shape: (batch_size, 1)

        # Assuming only the first instrument is being hedged
        zeros = tf.zeros((instrument_paths.shape[0], instrument_paths.shape[1] - 1), dtype=tf.float32)
        actions = tf.concat([action, zeros], axis=1)  # Shape: (batch_size, n_instruments)

        return actions

    def reset_last_delta(self, batch_size):
        """
        Reset the last delta to zero for each simulation in the batch.

        Arguments:
        - batch_size (int): The number of simulations in the batch.
        """
        self.last_delta = tf.zeros((batch_size,), dtype=tf.float32)

    def compute_deltas(self, S_values, T_minus_t_values):
        """
        Compute the deltas for a batch of stock prices and times to maturity.

        Arguments:
        - S_values (np.ndarray): Current stock prices. Shape: (batch_size,)
        - T_minus_t_values (np.ndarray): Time to maturity. Shape: (batch_size,)

        Returns:
        - deltas (np.ndarray): The computed deltas. Shape: (batch_size,)
        """
        deltas = []
        for S, T in zip(S_values, T_minus_t_values):
            # Handle cases where T <= 0
            if T <= 0:
                deltas.append(0.0)
                continue

            # Update the stock model's parameters
            original_S0 = self.stock_model.S0
            original_T = self.pricer.T

            self.stock_model.S0 = S
            self.pricer.T = T
            self.pricer.stock_model.S0 = S
            self.pricer.T = T
            self.geometric_agent.stock_model.S0 = S
            self.geometric_agent.T = T

            # Compute delta using central finite differences
            delta = self.delta(bump_size=self.bump_size)

            deltas.append(delta)

            # Restore original parameters
            self.stock_model.S0 = original_S0
            self.pricer.T = original_T
            self.pricer.stock_model.S0 = original_S0
            self.pricer.T = original_T
            self.geometric_agent.stock_model.S0 = original_S0
            self.geometric_agent.T = original_T

        return np.array(deltas, dtype=np.float32)

    def get_model_price(self):
        """
        Calculate the price of the arithmetic Asian option using the control variate method.

        Returns:
            float: The price of the option.
        """
        return self.price()
