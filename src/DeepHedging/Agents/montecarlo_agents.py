import tensorflow as tf
import numpy as np
from DeepHedging.Agents import BaseAgent

class MonteCarloBaseAgent(BaseAgent):
    """
    A base agent for Asian options using Monte Carlo pricing.

    This class provides common functionalities for pricing and delta hedging
    arithmetic and geometric Asian options using a Monte Carlo pricer.
    """

    plot_color = 'grey' 
    is_trainable = False

    def __init__(self, stock_model, option_class, r, T, num_simulations=10000, bump_size=0.01, seed=42):
        """
        Initialize the agent with market and option parameters.

        Arguments:
        - stock_model (Stock): An instance of a Stock subclass (e.g., GBMStock).
        - option_class (ContingentClaim): An instance of a ContingentClaim subclass.
        - r (float): Risk-free interest rate.
        - T (float): Time to maturity in years.
        - num_simulations (int): Number of Monte Carlo simulations.
        - bump_size (float): Relative size of the bump for finite differences.
        - seed (int): Random seed for reproducibility.
        """
        self.stock_model = stock_model
        self.option_class = option_class
        self.r = r
        self.T = T
        self.num_simulations = num_simulations
        self.bump_size = bump_size
        self.seed = seed

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

    def build_model(self):
        """
        No neural network model is needed for this agent.
        """
        pass

    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the Monte Carlo delta hedging strategy.

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

        # Compute delta using Monte Carlo pricer
        delta = tf.numpy_function(
            self.compute_deltas,
            [instrument_paths[:, 0].numpy(), T_minus_t.numpy()],
            tf.float32
        )

        # Calculate action as change in delta
        action = delta - self.last_delta
        self.last_delta = delta

        # Expand dimensions to match the number of instruments
        action = tf.expand_dims(action, axis=-1)  # Shape: (batch_size, 1)

        # Assuming only the first instrument is being hedged
        zeros = tf.zeros((instrument_paths.shape[0], instrument_paths.shape[1] - 1), dtype=tf.float32)
        actions = tf.concat([action, zeros], axis=1)  # Shape: (batch_size, n_instruments)

        return actions

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

            # Re-instantiate the pricer with updated T
            pricer = MonteCarloPricer(
                stock_model=self.stock_model,
                r=self.r,
                T=T,
                num_simulations=self.num_simulations,
                seed=self.seed
            )

            # Compute Delta
            delta = pricer.delta(contingent_claim=self.option_class)

            deltas.append(delta)

            # Restore original parameters
            self.stock_model.S0 = original_S0
            self.pricer.T = original_T

        return np.array(deltas, dtype=np.float32)


class ArithmeticAsianMonteCarloAgent(MonteCarloBaseAgent):
    """
    An agent that prices and hedges arithmetic Asian options using Monte Carlo simulation.
    """

    plot_color = 'grey' 
    name = 'asian_arithmetic_monte_carlo'
    plot_name = {
        'en': 'Asian Arithmetic Monte Carlo Delta',
        'es': 'Delta de Opción Asiática Aritmética - Monte Carlo'
    }

    def __init__(self, stock_model, option_class, r, T, num_simulations=10000, bump_size=0.01, seed=42):
        """
        Initialize the arithmetic Asian Monte Carlo agent.

        Arguments:
        - stock_model (Stock): An instance of a Stock subclass (e.g., GBMStock).
        - option_class (AsianArithmeticCall or AsianArithmeticPut): An instance of the option class.
        - r (float): Risk-free interest rate.
        - T (float): Time to maturity in years.
        - num_simulations (int): Number of Monte Carlo simulations.
        - bump_size (float): Relative size of the bump for finite differences.
        - seed (int): Random seed for reproducibility.
        """
        super().__init__(
            stock_model=stock_model,
            option_class=option_class,
            r=r,
            T=T,
            num_simulations=num_simulations,
            bump_size=bump_size,
            seed=seed
        )


class GeometricAsianMonteCarloAgent(MonteCarloBaseAgent):
    """
    An agent that prices and hedges geometric Asian options using Monte Carlo simulation.
    """

    plot_color = 'grey' 
    name = 'asian_geometric_monte_carlo'
    plot_name = {
        'en': 'Asian Geometric Monte Carlo Delta',
        'es': 'Delta de Opción Asiática Geométrica - Monte Carlo'
    }

    def __init__(self, stock_model, option_class, r, T, num_simulations=10000, bump_size=0.01, seed=42):
        """
        Initialize the geometric Asian Monte Carlo agent.

        Arguments:
        - stock_model (Stock): An instance of a Stock subclass (e.g., GBMStock).
        - option_class (AsianGeometricCall or AsianGeometricPut): An instance of the option class.
        - r (float): Risk-free interest rate.
        - T (float): Time to maturity in years.
        - num_simulations (int): Number of Monte Carlo simulations.
        - bump_size (float): Relative size of the bump for finite differences.
        - seed (int): Random seed for reproducibility.
        """
        super().__init__(
            stock_model=stock_model,
            option_class=option_class,
            r=r,
            T=T,
            num_simulations=num_simulations,
            bump_size=bump_size,
            seed=seed
        )


