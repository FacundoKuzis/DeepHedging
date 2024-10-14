import tensorflow as tf
import QuantLib as ql
import numpy as np
from DeepHedging.Agents import BaseAgent

class QuantlibAsianGeometricAgent(BaseAgent):
    """
    An agent that uses QuantLib to compute the delta hedging strategy for continuous geometric Asian options.
    """

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
        self.name = 'quantlib_asian_geometric_continuous'
        self.plot_name = 'QuantLib Asian Geometric Continuous Delta'

        # Initialize last delta for delta hedging
        self.last_delta = None

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
        delta = self.compute_delta_quantlib(instrument_paths[:, 0], T_minus_t)

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
        Compute the delta of the geometric Asian option using QuantLib.

        Arguments:
        - S_values (tf.Tensor): Current stock prices. Shape: (batch_size,)
        - T_minus_t_values (tf.Tensor): Time to maturity. Shape: (batch_size,)

        Returns:
        - deltas (tf.Tensor): The computed deltas. Shape: (batch_size,)
        """
        # Use tf.map_fn to apply quantlib_delta over the batch
        inputs = (S_values, T_minus_t_values)
        deltas = tf.map_fn(
            lambda x: tf.py_function(
                self.quantlib_delta, [x[0], x[1]], tf.float32
            ),
            elems=inputs,
            fn_output_signature=tf.float32
        )
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
        spot_price = float(spot_price)
        time_to_maturity = float(time_to_maturity)

        # Set up QuantLib parameters
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        settlement_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = settlement_date
        maturity_date = settlement_date + int(time_to_maturity * 365)

        # Option type
        if self.option_type.lower() == 'call':
            ql_option_type = ql.Option.Call
        elif self.option_type.lower() == 'put':
            ql_option_type = ql.Option.Put
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        # Payoff and exercise
        payoff = ql.PlainVanillaPayoff(ql_option_type, self.strike)
        exercise = ql.EuropeanExercise(maturity_date)

        # Average type
        average_type = ql.Average.Geometric

        # Asian option (continuous averaging)
        option = ql.ContinuousAveragingAsianOption(
            average_type,
            payoff,
            exercise
        )

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(settlement_date, self.r, day_count))
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(settlement_date, 0.0, day_count))
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(settlement_date, calendar, self.sigma, day_count))

        # Black-Scholes-Merton process
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_ts, flat_ts, vol_ts)

        # Pricing engine for continuous geometric Asian option
        engine = ql.AnalyticContinuousGeometricAveragePriceAsianEngine(bsm_process)
        option.setPricingEngine(engine)

        # Compute delta
        delta = option.delta()
        return np.float32(delta)

    def get_model_price(self):
        """
        Calculate the option price using QuantLib.

        Returns:
        - price (float): The QuantLib price of the option.
        """
        # Set up QuantLib parameters
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        settlement_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = settlement_date
        maturity_date = settlement_date + int(self.T * 365)

        # Option type
        if self.option_type.lower() == 'call':
            ql_option_type = ql.Option.Call
        elif self.option_type.lower() == 'put':
            ql_option_type = ql.Option.Put
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        # Payoff and exercise
        payoff = ql.PlainVanillaPayoff(ql_option_type, self.strike)
        exercise = ql.EuropeanExercise(maturity_date)

        # Average type
        average_type = ql.Average.Geometric

        # Asian option (continuous averaging)
        option = ql.ContinuousAveragingAsianOption(
            average_type,
            payoff,
            exercise
        )

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.S0))
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(settlement_date, self.r, day_count))
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(settlement_date, 0.0, day_count))
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(settlement_date, calendar, self.sigma, day_count))

        # Black-Scholes-Merton process
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_ts, flat_ts, vol_ts)

        # Pricing engine for continuous geometric Asian option
        engine = ql.AnalyticContinuousGeometricAveragePriceAsianEngine(bsm_process)
        option.setPricingEngine(engine)

        # Compute and return price
        price = option.NPV()
        return price