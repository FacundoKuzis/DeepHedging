import tensorflow as tf
import QuantLib as ql
import numpy as np
from DeepHedging.Agents import BaseAgent

class ArithmeticAsianMonteCarloAgent(BaseAgent):
    """
    An agent that uses QuantLib to compute the delta hedging strategy for arithmetic Asian options using numerical methods.
    """

    plot_color = 'grey' 
    name = 'asian_arithmetic_monte_carlo'
    is_trainable = False
    plot_name = {
        'en': 'Asian Arithmetic Monte Carlo Delta',
        'es': 'Delta de Opción Asiática Aritmética - Monte Carlo'
    }
    
    def __init__(self, gbm_stock, option_class):
        """
        Initialize the agent with market and option parameters.

        Arguments:
        - gbm_stock (GBMStock): An instance containing the stock parameters.
        - option_class: An instance of the option class containing option parameters.
        """
        self.S0 = gbm_stock.S0
        self.T = gbm_stock.T
        self.N = gbm_stock.N
        self.r = gbm_stock.r
        self.sigma = gbm_stock.sigma
        self.strike = option_class.strike
        self.option_type = option_class.option_type
        self.dt = gbm_stock.dt

        # Initialize last delta for delta hedging
        self.last_delta = None

        # Pre-initialize QuantLib objects
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.NullCalendar()
        self.settlement_date = ql.Date.todaysDate()
        self.maturity_date = self.settlement_date + int(self.T * 365 + 0.5)

        # Option type
        if self.option_type.lower() == 'call':
            self.ql_option_type = ql.Option.Call
        elif self.option_type.lower() == 'put':
            self.ql_option_type = ql.Option.Put
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        # Payoff and exercise
        self.payoff = ql.PlainVanillaPayoff(self.ql_option_type, self.strike)
        self.exercise = ql.EuropeanExercise(self.maturity_date)

        # Average type
        self.average_type = ql.Average.Arithmetic

        # Averaging dates
        self.N_time_steps = self.N
        delta_time = self.T / self.N_time_steps
        self.averaging_dates = [self.settlement_date + int((i+1)*delta_time*365 + 0.5) for i in range(self.N_time_steps)]
        self.averaging_dates[-1] = self.maturity_date  # Ensure last date is maturity

        # Asian option (discrete arithmetic averaging)
        self.option = ql.DiscreteAveragingAsianOption(
            self.average_type,
            0.0,  # running sum (no past fixings)
            0,    # past fixings
            self.averaging_dates,
            self.payoff,
            self.exercise
        )

        # Market data
        self.spot_quote = ql.SimpleQuote(self.S0)
        self.spot_handle = ql.QuoteHandle(self.spot_quote)
        self.flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.settlement_date, self.r, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.settlement_date, 0.0, self.day_count))
        self.vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.settlement_date, self.calendar, self.sigma, self.day_count))

        # Black-Scholes-Merton process
        self.bsm_process = ql.BlackScholesMertonProcess(
            self.spot_handle, self.dividend_ts, self.flat_ts, self.vol_ts)

        # Monte Carlo engine parameters
        rng = 'pseudorandom'
        antithetic = True
        required_samples = 10000
        seed = 42

        # Pricing engine for discrete arithmetic Asian option using Monte Carlo
        self.mc_engine = ql.MCDiscreteArithmeticAPEngine(
            self.bsm_process,
            rng,
            False,       # brownianBridge
            antithetic,  # antitheticVariate
            False,       # controlVariate
            required_samples,
            None,        # requiredTolerance
            None,        # maxSamples
            seed
        )
        self.option.setPricingEngine(self.mc_engine)

    def build_model(self):
        """
        No neural network model is needed for this agent.
        """
        pass

    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the QuantLib delta hedging strategy.

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

        # Compute delta using QuantLib
        delta = tf.numpy_function(
            self.compute_delta_quantlib,
            [instrument_paths[:, 0], T_minus_t],
            tf.float32
        )

        # Calculate action as change in delta
        action = delta - self.last_delta
        self.last_delta = delta

        # Expand dimensions to match the number of instruments
        action = tf.expand_dims(action, axis=-1)  # Shape: (batch_size, 1)

        # Assuming only the first instrument is being hedged
        zeros = tf.zeros((instrument_paths.shape[0], instrument_paths.shape[1] - 1))
        actions = tf.concat([action, zeros], axis=1)  # Shape: (batch_size, n_instruments)

        return actions

    def reset_last_delta(self, batch_size):
        """
        Reset the last delta to zero for each simulation in the batch.

        Arguments:
        - batch_size (int): The number of simulations in the batch.
        """
        self.last_delta = tf.zeros((batch_size,), dtype=tf.float32)

    def process_batch(self, batch_paths, batch_T_minus_t):
        """
        Process a batch of paths to compute the hedging actions at each timestep.

        Arguments:
        - batch_paths (tf.Tensor): Paths of the instruments. Shape: (batch_size, timesteps, n_instruments)
        - batch_T_minus_t (tf.Tensor): Time to maturity at each timestep. Shape: (batch_size, timesteps)

        Returns:
        - all_actions (tf.Tensor): Hedging actions at each timestep. Shape: (batch_size, timesteps, n_instruments)
        """
        self.reset_last_delta(batch_paths.shape[0])
        all_actions = []
        for t in range(batch_paths.shape[1] - 1):  # Exclude the terminal timestep
            print(t)
            current_paths = batch_paths[:, t, :]       # Shape: (batch_size, n_instruments)
            current_T_minus_t = batch_T_minus_t[:, t]  # Shape: (batch_size,)
            action = self.act(current_paths, current_T_minus_t)
            all_actions.append(action)

        # Stack actions and append zero action for the terminal timestep
        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[-1]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)

        return all_actions  # Shape: (batch_size, timesteps, n_instruments)

    def compute_delta_quantlib(self, S_values, T_minus_t_values):
        """
        Compute the delta of the arithmetic Asian option using QuantLib.

        Arguments:
        - S_values (np.ndarray): Current stock prices. Shape: (batch_size,)
        - T_minus_t_values (np.ndarray): Time to maturity. Shape: (batch_size,)

        Returns:
        - deltas (np.ndarray): The computed deltas. Shape: (batch_size,)
        """
        S_values = np.array(S_values, dtype=np.float64)
        T_minus_t_values = np.array(T_minus_t_values, dtype=np.float64)

        batch_size = len(S_values)
        deltas = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            spot_price = S_values[i]
            time_to_maturity = T_minus_t_values[i]
            delta = self.quantlib_delta(spot_price, time_to_maturity)
            deltas[i] = delta

        return deltas

    def quantlib_delta(self, spot_price, time_to_maturity):
        """
        Use QuantLib to compute the delta for a single instance.

        Arguments:
        - spot_price (float): The current stock price.
        - time_to_maturity (float): Time to maturity.

        Returns:
        - delta (float): The computed delta.
        """
        # Update spot price
        self.spot_quote.setValue(spot_price)

        # Update evaluation date
        evaluation_date = self.maturity_date - int(time_to_maturity * 365 + 0.5)
        ql.Settings.instance().evaluationDate = evaluation_date

        # Adjust yield curve and volatility term structures if necessary
        # (Assuming constant rates and volatilities, so no need to adjust)

        # Recalculate the option price
        epsilon = spot_price * 0.01  # Bump size (1% of spot price)

        # Price at spot_price + epsilon
        self.spot_quote.setValue(spot_price + epsilon)
        up_price = self.option.NPV()

        # Price at spot_price
        self.spot_quote.setValue(spot_price)
        down_price = self.option.NPV()

        # Central difference approximation of delta
        delta = (up_price - down_price) / (epsilon)

        return delta

    def get_model_price(self):
        """
        Calculate the option price using QuantLib.

        Returns:
        - price (float): The QuantLib price of the option.
        """
        # Recalculate the option price
        price = self.option.NPV()
        return price
