import tensorflow as tf
import tensorflow_probability as tfp
from DeepHedging.Agents import BaseAgent

class DeltaHedgingAgent(BaseAgent):
    """
    A delta hedging agent that computes the delta of an option and uses it as the hedging strategy.

    Arguments:
    - stock_model (GBMStock): An instance of the GBMStock class containing the stock parameters.
    - strike (float): Strike price of the option.
    - option_type (str): Type of the option ('call' or 'put').
    """

    plot_color = 'orange' 
    name = 'bs_delta_hedging'
    is_trainable = False
    plot_name = {
        'en': 'BS European Vanilla Delta',
        'es': 'Delta de Opción Europea Vanilla'
    }

    def __init__(self, stock_model, option_class):
        self.stock_model = stock_model
        self.S0 = stock_model.S0
        self.T = stock_model.T
        self.N = stock_model.N
        self.r = stock_model.r
        self.sigma = stock_model.sigma
        self.strike = option_class.strike
        self.option_type = option_class.option_type
        self.dt = stock_model.dt

    def build_model(self):
        """
        Dummy implementation as no model building is required for delta hedging.
        """
        pass

    def d1(self, S, T_minus_t):
        """
        Calculate the d1 component used in the Black-Scholes formula.

        Arguments:
        - S (tf.Tensor): The current stock price.
        - T_minus_t (tf.Tensor): The current T - t.

        Returns:
        - d1 (tf.Tensor): The d1 value.
        """

        eps = 1e-4
        return (tf.math.log(S / self.strike) + (self.r + 0.5 * self.sigma ** 2) * (T_minus_t + eps)) / (self.sigma * tf.sqrt(T_minus_t + eps))

    def delta(self, S, T_minus_t):
        """
        Calculate the delta of the option.

        Arguments:
        - S (tf.Tensor): The current stock price.
        - t (tf.Tensor): The current time.

        Returns:
        - delta (tf.Tensor): The delta value.
        """
        d1 = self.d1(S, T_minus_t)
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        if self.option_type == 'call':
            return normal_dist.cdf(d1)
        elif self.option_type == 'put':
            return normal_dist.cdf(d1) - 1.0
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the delta hedging strategy.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The delta value used as the hedging action.
        """

        delta = self.delta(instrument_paths[:, 0], T_minus_t) # ASSUMPTION: Stock is the first instrument
        action = delta - self.last_delta
        self.last_delta = delta
        #actions_expanded = tf.expand_dims(action, axis=-1)
        #zeros = tf.zeros_like(instrument_paths)
        action =  tf.expand_dims(action, axis=-1)
        zeros = tf.zeros((instrument_paths.shape[0], instrument_paths.shape[1]-1))
        actions = tf.concat([action, zeros], axis=1)
        return actions

    def reset_last_delta(self, batch_size):
        self.last_delta = tf.zeros((batch_size,), dtype=tf.float32)

    def process_batch(self, batch_paths, batch_T_minus_t):
        self.reset_last_delta(batch_paths.shape[0])
        all_actions = []
        for t in range(batch_paths.shape[1] -1):  # timesteps until T-1
            print(t)
            current_paths = batch_paths[:, t, :] # (n_simulations, n_timesteps, n_instruments)
            current_T_minus_t = batch_T_minus_t[:, t] # (n_simulations, n_timesteps)
            action = self.act(current_paths, current_T_minus_t)
            all_actions.append(action)

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[-1]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)

        return all_actions # (n_simulations, n_timesteps, n_instruments)

    def get_model_price(self):
        """
        Calculate the Black-Scholes price for the option.

        Returns:
        - price (tf.Tensor): The Black-Scholes price of the option.
        """
        d1 = self.d1(self.S0, self.T)
        d2 = d1 - self.sigma * tf.sqrt(self.T)
        
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        if self.option_type == 'call':
            price = (self.S0 * normal_dist.cdf(d1) - 
                     self.strike * tf.exp(-self.r * self.T) * normal_dist.cdf(d2))
        elif self.option_type == 'put':
            price = (self.strike * tf.exp(-self.r * self.T) * normal_dist.cdf(-d2) - 
                     self.S0 * normal_dist.cdf(-d1))
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")
        
        return price