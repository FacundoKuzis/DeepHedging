import tensorflow as tf
import numpy as np
from DeepHedging.Agents import DeltaHedgingAgent
from DeepHedging.utils import MonteCarloPricer

class MonteCarloAgent(DeltaHedgingAgent):
    """
    A base agent for Asian options using Monte Carlo pricing.

    This class provides common functionalities for pricing and delta hedging
    arithmetic and geometric Asian options using a Monte Carlo pricer.
    """

    plot_color = 'grey' 
    is_trainable = False
    name = 'montecarlo'
    plot_name = {
        'en': 'Monte Carlo Delta',
        'es': 'Delta calculada con Monte Carlo'
    }

    def __init__(
        self,
        stock_model,
        option_class,
        num_simulations=10000,
        bump_size=0.01,
        seed=33,
        no_trade_band=0.0,
        mc_use_vectorized=True,
        mc_chunk_size=64,
        mc_parallel_enabled=False,
        mc_n_workers=1,
        mc_parallel_backend="thread",
    ):
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
        super().__init__(stock_model, option_class, no_trade_band=no_trade_band)
        self.stock_model = stock_model
        self.option_class = option_class
        self.num_simulations = num_simulations
        self.bump_size = bump_size
        self.seed = seed

        # Instantiate Monte Carlo Pricer
        self.pricer = MonteCarloPricer(
            stock_model=self.stock_model,
            r=self.r,
            T=self.T,
            num_simulations=self.num_simulations,
            seed=self.seed,
            use_vectorized=bool(mc_use_vectorized),
            chunk_size=int(mc_chunk_size),
            parallel_enabled=bool(mc_parallel_enabled),
            n_workers=int(mc_n_workers),
            parallel_backend=str(mc_parallel_backend),
        )
        self._mc_profile_rows = []

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
            [instrument_paths[:, 0], T_minus_t],
            tf.float32
        )
        delta.set_shape((instrument_paths.shape[0],))

        return self._to_actions(delta, instrument_paths)

    def compute_deltas(self, S_values, T_minus_t_values):
        """
        Compute the deltas for a batch of stock prices and times to maturity.

        Arguments:
        - S_values (np.ndarray): Current stock prices. Shape: (batch_size,)
        - T_minus_t_values (np.ndarray): Time to maturity. Shape: (batch_size,)

        Returns:
        - deltas (np.ndarray): The computed deltas. Shape: (batch_size,)
        """
        S_values = np.asarray(S_values, dtype=np.float64).reshape(-1)
        T_minus_t_values = np.asarray(T_minus_t_values, dtype=np.float64).reshape(-1)

        deltas = np.zeros_like(S_values, dtype=np.float32)
        alive_mask = T_minus_t_values > 0.0
        if not np.any(alive_mask):
            return deltas.astype(np.float32)

        alive_idx = np.where(alive_mask)[0]
        alive_S = S_values[alive_mask]
        alive_T = T_minus_t_values[alive_mask]
        deltas_alive, profile_row = self.pricer.delta_batch_with_S0(
            contingent_claim=self.option_class,
            S0_values=alive_S,
            T_values=alive_T,
            bump_size=self.bump_size,
            use_common_random_numbers=True,
            seed=self.seed,
            profile=True,
        )
        self._mc_profile_rows.append(profile_row)
        deltas[alive_idx] = deltas_alive.astype(np.float32)
        return deltas.astype(np.float32)

    
    def get_model_price(self):
        return self.pricer.price_with_S0(self.option_class, self.S0)

