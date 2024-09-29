import tensorflow as tf
import tensorflow_probability as tfp
from DeepHedging.Agents import DeltaHedgingAgent
from DeepHedging.ContingentClaims import EuropeanCall, EuropeanPut

class ChooserDeltaHedgingAgent(DeltaHedgingAgent):
    """
    A delta hedging agent that computes the delta of a chooser option and uses it as the hedging strategy.

    Arguments:
    - gbm_stock (GBMStock): An instance of the GBMStock class containing the stock parameters.
    - chooser_option (ChooserOption): An instance of the ChooserOption class containing the option parameters.
    - t_choice (float): Time until the choice date from inception.
    """

    def __init__(self, gbm_stock, chooser_option, t_choice):
        super().__init__(gbm_stock, chooser_option)
        self.t_choice = t_choice
        self.q = getattr(gbm_stock, 'q', 0.0)
        self.name = 'delta_hedging_chooser'
        self.plot_name = 'Chooser Option Delta'

        self.call_delta_agent = DeltaHedgingAgent(gbm_stock, EuropeanCall(chooser_option.strike,
                                                                          amount=chooser_option.amount))
        
        self.put_delta_agent = DeltaHedgingAgent(gbm_stock, EuropeanPut(chooser_option.strike,
                                                                          amount=chooser_option.amount))

        self.converted_option_type = None

    def delta(self, S, T_minus_t):
        """
        Calculate the delta of the chooser option, handling conversion after the choice date.

        Arguments:
        - S (tf.Tensor): The current stock prices (batch_size,).
        - T_minus_t (tf.Tensor): The time to maturity at current time t (batch_size,).

        Returns:
        - delta (tf.Tensor): The delta values (batch_size,).
        """
        eps = 1e-6
        # Assuming t is same for all samples
        t = self.T - T_minus_t[0]  # Current time since inception

        t_choice_minus_t = self.t_choice - t  # Time until the choice date

        # Determine if we are before or after the choice date
        is_after_choice = t_choice_minus_t <= eps 

        if is_after_choice:  # After or at the choice date
            need_conversion = self.converted_option_type is None
            if need_conversion:
                self.convert_options(S, T_minus_t)

            # Determine which samples are calls and which are puts
            is_call = tf.equal(self.converted_option_type, 'call')  # (batch_size,)
            mask_call = tf.cast(is_call, tf.bool)
            mask_put = tf.logical_not(mask_call)

            # Compute delta only for calls
            S_call = tf.boolean_mask(S, mask_call)                 # (num_calls,)
            T_call = tf.boolean_mask(T_minus_t, mask_call)         # (num_calls,)
            delta_call = self.call_delta_agent.delta(S_call, T_call)  # (num_calls,)

            # Compute delta only for puts
            S_put = tf.boolean_mask(S, mask_put)                   # (num_puts,)
            T_put = tf.boolean_mask(T_minus_t, mask_put)           # (num_puts,)
            delta_put = self.put_delta_agent.delta(S_put, T_put)      # (num_puts,)

            # Initialize a delta tensor with zeros
            delta = tf.zeros_like(S, dtype=delta_call.dtype)

            # Scatter the computed deltas back to their original positions
            delta = tf.tensor_scatter_nd_update(
                delta,
                tf.where(mask_call),
                delta_call
            )
            delta = tf.tensor_scatter_nd_update(
                delta,
                tf.where(mask_put),
                delta_put
            )
        else:
            # Before the choice date
            delta = self.delta_chooser(S, T_minus_t, t_choice_minus_t)

        return delta


    def delta_chooser(self, S, T_minus_t, t_choice_minus_t):
        """
        Calculate the delta of the chooser option before the choice date.

        Arguments:
        - S (tf.Tensor): The current stock prices (batch_size,).
        - T_minus_t (tf.Tensor): The time to maturity at current time t (batch_size,).
        - t_choice_minus_t (float): The time until the choice date.

        Returns:
        - delta (tf.Tensor): The delta values (batch_size,).
        """
        eps = 1e-6
        T_minus_t = tf.maximum(T_minus_t, eps)
        t_choice_minus_t = tf.maximum(t_choice_minus_t, eps)  # Scalar

        ln_S_over_X = tf.math.log(S / self.strike)
        d1 = (ln_S_over_X + (self.r - self.q + 0.5 * self.sigma ** 2) * T_minus_t) / (
            self.sigma * tf.sqrt(T_minus_t)
        )
        d1_star = (ln_S_over_X + (self.r - self.q) * T_minus_t + 0.5 * self.sigma ** 2 * t_choice_minus_t) / (
            self.sigma * tf.sqrt(t_choice_minus_t)
        )

        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        N_d1 = normal_dist.cdf(d1)
        N_d1_star = normal_dist.cdf(d1_star)

        delta = tf.exp(-self.q * T_minus_t) * (N_d1 + N_d1_star - 1)
        return delta

    def convert_options(self, S, T_minus_t):
        """
        Convert the chooser option into a call or put after the choice date.

        Arguments:
        - S (tf.Tensor): The current stock prices (batch_size,).
        - T_minus_t (tf.Tensor): The time to maturity at current time t (batch_size,).
        """
        # Compute option prices for these samples
        call_prices = self.call_delta_agent.get_model_price(S, T_minus_t)
        put_prices = self.put_delta_agent.get_model_price(S, T_minus_t)

        # Determine which option is more valuable per sample
        is_call_more_valuable = call_prices > put_prices

        # Use tf.where with scalar string tensors, which will be broadcasted
        converted_types = tf.where(
            is_call_more_valuable,
            tf.constant('call', dtype=tf.string),
            tf.constant('put', dtype=tf.string)
        )

        # Update the converted_option_type tensor
        self.converted_option_type = converted_types


    def get_option_price(self, S, T_minus_t):
        """
        Calculate the price of the chooser option.

        Arguments:
        - S (tf.Tensor): The current stock prices (batch_size,).
        - T_minus_t (tf.Tensor): The time to maturity at current time t (batch_size,).

        Returns:
        - price (tf.Tensor): The option prices (batch_size,).
        """
        raise NotImplementedError("Chooser option pricing is not implemented.")

    def get_model_price(self, S = None, T_minus_t = None):
        """
        Calculate the price of the chooser option.

        Arguments:
        - S (tf.Tensor): The current stock prices (batch_size,).
        - T_minus_t (tf.Tensor): The time to maturity at current time t (batch_size,).

        Returns:
        - price (tf.Tensor): The option prices (batch_size,).
        """
        eps = 1e-6
        if S is None:
            S = tf.constant(self.S0, dtype=tf.float32)
        if T_minus_t is None:
            T_minus_t = tf.constant(self.T, dtype=tf.float32)

        T_minus_t = tf.maximum(T_minus_t, eps)
        t = self.T - T_minus_t  # Current time since inception
        t_choice_minus_t = self.t_choice - t  # Time until the choice date
        t_choice_minus_t = tf.maximum(t_choice_minus_t, eps)  # Ensure positive

        # Calculate d1 and d2
        sqrt_T_minus_t = tf.sqrt(T_minus_t)
        ln_S_over_X = tf.math.log(S / self.strike)
        d1 = (ln_S_over_X + (self.r - self.q + 0.5 * self.sigma ** 2) * T_minus_t) / (self.sigma * sqrt_T_minus_t)
        d2 = d1 - self.sigma * sqrt_T_minus_t

        # Calculate d1_star and d2_star
        sqrt_t_choice_minus_t = tf.sqrt(t_choice_minus_t)
        d1_star = (ln_S_over_X + (self.r - self.q) * T_minus_t + 0.5 * self.sigma ** 2 * t_choice_minus_t) / (
            self.sigma * sqrt_t_choice_minus_t
        )
        d2_star = d1_star - self.sigma * sqrt_t_choice_minus_t

        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        N_d1 = normal_dist.cdf(d1)
        N_d2 = normal_dist.cdf(d2)
        N_minus_d1_star = normal_dist.cdf(-d1_star)
        N_minus_d2_star = normal_dist.cdf(-d2_star)

        # Calculate the chooser option price using the provided formula
        price = (
            S * tf.exp(-self.q * T_minus_t) * N_d1
            - self.strike * tf.exp(-self.r * T_minus_t) * N_d2
            - S * tf.exp(-self.q * T_minus_t) * N_minus_d1_star
            + self.strike * tf.exp(-self.r * T_minus_t) * N_minus_d2_star
        )

        return price
