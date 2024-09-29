import tensorflow as tf
import os
import time
from abc import ABC, abstractmethod
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings

import tensorflow as tf
from abc import ABC, abstractmethod
import os
import warnings

class BaseAgent(ABC):
    """
    The base class for all agents.

    Methods:
    - build_model(self): Abstract method to build the model architecture.
    - act(self, instrument_paths, T_minus_t): Abstract method to act based on inputs.
    - transform_input(self, *args): Optional method to transform inputs if necessary.
    """

    @abstractmethod
    def build_model(self):
        pass

    def transform_paths(self, instrument_paths, transformation_configs=None):
        """
        Transforms the instrument paths based on the specified transformation configurations.

        Arguments:
        - instrument_paths (tf.Tensor): The paths of the instrument. Shape: (batch_size, n_timesteps, n_instruments)
        - transformation_configs (list of dict, optional): List of dictionaries, each containing:
        - 'transformation_type' (str): The type of transformation ('log', 'log_moneyness').
        - 'K' (float, optional): The strike price, required if transformation_type is 'log_moneyness'.

        Returns:
        - transformed_paths (tf.Tensor): The transformed instrument paths. Shape: (batch_size, n_timesteps, n_instruments)
        """

        if transformation_configs is None:
            # No transformation, return the paths as-is
            return instrument_paths

        # Ensure the length of the list matches the number of instruments
        if len(transformation_configs) != instrument_paths.shape[-1]:
            raise ValueError("Length of transformation_configs list must match the number of instruments.")

        transformed_paths_list = []
        for i, config in enumerate(transformation_configs):
            if len(instrument_paths.shape) == 3:
                path_i = instrument_paths[:, :, i]  # Extract the path for the i-th instrument
            else:
                path_i = instrument_paths[:, i]  # Extract the path for the i-th instrument

            t_type = config.get('transformation_type')
            K = config.get('K')
            transformed_path_i = self._apply_transformation(path_i, t_type, K)
            transformed_paths_list.append(transformed_path_i)

        # Stack the transformed paths back into a single tensor
        transformed_paths = tf.stack(transformed_paths_list, axis=-1)
        return transformed_paths

    def _apply_transformation(self, instrument_paths, transformation_type, K=None):
        """
        Apply the specified transformation to the given instrument paths.

        Arguments:
        - instrument_paths (tf.Tensor): The paths of the instrument.
        - transformation_type (str or None): The type of transformation ('log', 'log_moneyness').
        - K (float, optional): The strike price, required if transformation_type is 'log_moneyness'.

        Returns:
        - transformed_paths (tf.Tensor): The transformed instrument paths.
        """
        if transformation_type is None:
            return instrument_paths
        elif transformation_type == 'log':
            return tf.math.log(instrument_paths)
        elif transformation_type == 'log_moneyness':
            if K is None:
                raise ValueError("Strike price K must be provided for 'log moneyness' transformation.")
            return tf.math.log(instrument_paths / K)
        else:
            raise ValueError(f"Unsupported transformation type: {transformation_type}")

    def transform_input(self, instrument_paths, T_minus_t):
        """
        Transforms the input by concatenating instrument paths and time to maturity.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
                                        Shape: (batch_size, input_shape)
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.
                                Shape: (batch_size,)

        Returns:
        - input_data (tf.Tensor): The transformed input data.
                                Shape: (batch_size, input_shape + 1)
        """
        #instrument_paths = tf.expand_dims(instrument_paths, axis=-1)  # Shape: (n_instruments, batch_size, n_timesteps, 1)
        #T_minus_t = tf.expand_dims(T_minus_t, axis=-1)  # Shape: (batch_size, n_timesteps, 1)

        instrument_paths = self.transform_paths(instrument_paths, self.path_transformation_configs) # Shape: (batch_size, n_timesteps, n_instruments) or (batch_size, n_instruments)
        # T_minus_t  # Shape: (batch_size, n_timesteps)
        T_minus_t_expanded = tf.expand_dims(T_minus_t, axis=-1)  # Shape: (batch_size, n_timesteps, 1) or (batch_size, 1)

        # Concatenate along the last axis
        input_data = tf.concat([instrument_paths, T_minus_t_expanded], axis=-1) # Shape (batch_size, n_timesteps, n_instruments+1) or (batch_size, n_instruments+1)

        return input_data
    
    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the input.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The action chosen by the model.
        """
        input_data = self.transform_input(instrument_paths, T_minus_t)
        action = self.model(input_data)
        return action

    def train_batch(self, batch_paths, batch_T_minus_t, optimizer, loss_function):
        """
        Train the model on a batch of data, processing timestep by timestep.

        Arguments:
        - batch_paths (tf.Tensor): Tensor containing a batch of instrument paths. Shape: (batch_size, timesteps, input_shape)
        - batch_T_minus_t (tf.Tensor): Tensor containing the time to maturity at each timestep.
        - optimizer (tf.optimizers.Optimizer): The optimizer to use for training.

        Returns:
        - loss (tf.Tensor): The loss value after training on the batch.
        """
        with tf.GradientTape() as tape:
            actions = self.process_batch(batch_paths, batch_T_minus_t) # (batch_size, N+1, n_instruments)
            loss = loss_function(batch_paths, actions)

            # Compute gradients based on the total loss
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def load_model(self, model_path):
        """
        Load the model from the specified path.

        Arguments:
        - model_path (str): File path from which the model will be loaded.
        """
        if not os.path.exists(model_path):
            warnings.warn(f"Model path '{model_path}' does not exist. Exiting the load function.")
            return

        self.model = tf.keras.models.load_model(model_path)

    def save_model(self, model_path):
        """
        Save the model to the specified path.

        Arguments:
        - model_path (str): File path where the model will be saved.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

class DeltaHedgingAgent(BaseAgent):
    """
    A delta hedging agent that computes the delta of an option and uses it as the hedging strategy.

    Arguments:
    - gbm_stock (GBMStock): An instance of the GBMStock class containing the stock parameters.
    - strike (float): Strike price of the option.
    - option_type (str): Type of the option ('call' or 'put').
    """

    def __init__(self, gbm_stock, option_class):
        self.S0 = gbm_stock.S0
        self.T = gbm_stock.T
        self.N = gbm_stock.N
        self.r = gbm_stock.r
        self.sigma = gbm_stock.sigma
        self.strike = option_class.strike
        self.option_type = option_class.option_type
        self.dt = gbm_stock.dt
        self.name = 'delta_hedging'
        self.plot_name = 'BS Delta'

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
        if self.option_type in 'call':
            price = (self.S0 * normal_dist.cdf(d1) - 
                     self.strike * tf.exp(-self.r * self.T) * normal_dist.cdf(d2))
        elif self.option_type == 'put':
            price = (self.strike * tf.exp(-self.r * self.T) * normal_dist.cdf(-d2) - 
                     self.S0 * normal_dist.cdf(-d1))
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")
        
        return price

    def get_model_price(self, S=None, T_minus_t=None):
        """
        Calculate the Black-Scholes price for the option.

        Arguments:
        - S (tf.Tensor, optional): The current stock prices (batch_size,). If None, use self.S0.
        - T_minus_t (tf.Tensor, optional): The time to maturity at current time t (batch_size,). If None, use self.T.

        Returns:
        - price (tf.Tensor): The Black-Scholes price of the option (batch_size,).
        """

        if S is None:
            S = self.S0
        if T_minus_t is None:
            T = self.T
        else:
            T = T_minus_t  # Shape: (batch_size,)

        # Compute d1 and d2 using the Black-Scholes formula
        d1 = self.d1(S, T)  # Should handle tensor inputs
        d2 = d1 - self.sigma * tf.sqrt(T)

        # Define the standard normal distribution
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)

        # Calculate the option price based on the option type
        if self.option_type == 'call':
            price = (S * normal_dist.cdf(d1)) - (self.strike * tf.exp(-self.r * T) * normal_dist.cdf(d2))
        elif self.option_type == 'put':
            price = (self.strike * tf.exp(-self.r * T) *  normal_dist.cdf(-d2)) - (S * normal_dist.cdf(-d1))
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return price

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

class LSTMAgent(BaseAgent):
    """
    An LSTM agent that processes the entire sequence of inputs at once.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    def __init__(self, n_hedging_timesteps, path_transformation_configs = None,
                 n_instruments = 1):

        self.input_shape = (n_hedging_timesteps, n_instruments + 1) # +1 for T-t
        self.n_instruments = n_instruments
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.path_transformation_configs = path_transformation_configs
        self.name = 'lstm'
        self.plot_name = 'LSTM'

    def build_model(self, input_shape, output_shape):
        """
        Builds an LSTM model.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LSTM(30, return_sequences=True),
            tf.keras.layers.LSTM(30, return_sequences=True),
            tf.keras.layers.Dense(output_shape, activation='linear')
        ])
        return model

    def process_batch(self, batch_paths, batch_T_minus_t):

        all_actions = self.act(batch_paths[:, :-1, :], batch_T_minus_t) # (batch_size, N, n_instruments)

        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[2]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)
        #all_actions = all_actions[:, :, 0] # temporary, only one instrument

        return all_actions

class GRUAgent(LSTMAgent):
    """
    A GRU agent that processes the entire sequence of inputs at once.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    def __init__(self, n_hedging_timesteps, path_transformation_configs = None,
                 n_instruments = 1):
        
        super().__init__(n_hedging_timesteps, path_transformation_configs, 
                 n_instruments)
        self.name = 'gru'
        self.plot_name = 'GRU'

    def build_model(self, input_shape, output_shape):
        """
        Builds a GRU model.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.GRU(30, return_sequences=True),
            tf.keras.layers.GRU(30, return_sequences=True, activation = 'relu'),
            tf.keras.layers.Dense(output_shape, activation='linear')
        ])
        return model

class SimpleAgent(BaseAgent):
    """
    A simple agent that processes inputs timestep by timestep.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    def __init__(self, path_transformation_configs = None, n_instruments = 1):
        self.input_shape = (n_instruments + 1,) # +1 for T-t
        self.n_instruments = n_instruments
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.path_transformation_configs = path_transformation_configs
        self.name = 'simple'
        self.plot_name = 'Simple'

    def build_model(self, input_shape, output_shape):
        """
        Builds a simple feedforward model.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='linear')
        ])
        return model

    def process_batch(self, batch_paths, batch_T_minus_t):
        all_actions = []
        for t in range(batch_paths.shape[1] -1):  # timesteps until T-1
            current_paths = batch_paths[:, t, :] # (n_simulations, n_timesteps, n_instruments)
            current_T_minus_t = batch_T_minus_t[:, t] # (n_simulations, n_timesteps)
            action = self.act(current_paths, current_T_minus_t)
            all_actions.append(action)

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[-1]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)

        return all_actions # (n_simulations, n_timesteps, n_instruments)

class RecurrentAgent(SimpleAgent):
    """
    A recurrent agent that processes inputs timestep by timestep and includes the accumulated position.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    def __init__(self, path_transformation_configs = None, n_instruments = 1):
        self.input_shape = (n_instruments + 1 + n_instruments,) # +1 for T-t and + n_instruments for accumulated position
        self.n_instruments = n_instruments
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.accumulated_position = None  # Initialize accumulated position
        self.path_transformation_configs = path_transformation_configs
        self.name = 'recurrent'
        self.plot_name = 'Recurrent'

    def build_model(self, input_shape, output_shape):
        """
        Builds a feedforward model with additional input for accumulated position.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),  # Adjust input shape for accumulated position and T_minus_t
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='linear')
        ])
        return model

    def reset_accumulated_position(self, batch_size):
        """
        Resets the accumulated position to zero for each batch.

        Arguments:
        - batch_size (int): The size of the batch being processed.
        """
        self.accumulated_position = tf.zeros((batch_size, self.n_instruments), dtype=tf.float32)

    def transform_input(self, instrument_paths, T_minus_t):
        """
        Transforms the input by including the accumulated position.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - transformed_input (tf.Tensor): The transformed input, including accumulated position and T_minus_t.
        """
        # Concatenate the instrument paths, accumulated position, and T_minus_t
        input_data = super().transform_input(instrument_paths, T_minus_t) 
        transformed_input = tf.concat([input_data, self.accumulated_position], axis=-1)

        return transformed_input # (batch_size, n_instruments + 1 + n_instruments)


    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the input.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The action chosen by the model.
        """
        input_data = self.transform_input(instrument_paths, T_minus_t)
        action = self.model(input_data)

        # Update accumulated position by summing actions
        self.accumulated_position += action  # Accumulate positions across timesteps

        return action

    def process_batch(self, batch_paths, batch_T_minus_t):
        """
        Processes the entire batch timestep by timestep, updating the accumulated position at each step.

        Arguments:
        - batch_paths (tf.Tensor): Tensor containing a batch of instrument paths. Shape: (batch_size, timesteps, input_shape)
        - batch_T_minus_t (tf.Tensor): Tensor containing the time to maturity at each timestep.

        Returns:
        - all_actions (tf.Tensor): Tensor containing all the actions taken for the batch.
        """
        # Reset accumulated position at the start of processing each batch
        self.reset_accumulated_position(batch_paths.shape[0])
        return super().process_batch(batch_paths, batch_T_minus_t)

class WaveNetAgent(LSTMAgent):
    """
    A WaveNet agent that processes the entire sequence of inputs using a causal WaveNet-like architecture.

    Arguments:
    - n_hedging_timesteps (int): Number of timesteps in the hedging sequence.
    - path_transformation_type (str or None): Type of transformation applied to the instrument paths.
    - K (float or None): Strike price or other parameter for path transformation.
    - num_filters (int): Number of filters in the convolutional layers.
    - num_residual_blocks (int): Number of residual blocks in the WaveNet model.
    """

    def __init__(self, n_hedging_timesteps, path_transformation_configs = None, num_filters=32, num_residual_blocks=3,
                 n_instruments = 1):
        
        self.input_shape = (n_hedging_timesteps, n_instruments + 1) # +1 for T-t
        self.n_instruments = n_instruments
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.path_transformation_configs = path_transformation_configs
        self.name = 'wavenet'
        self.plot_name = 'WaveNet'

    def build_model(self, input_shape, output_shape):
        """
        Builds a causal WaveNet-like model.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        # Initial causal convolutional layer
        x = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=1, padding='causal', activation='relu')(x)

        # Residual blocks with dilated causal convolutions
        for i in range(self.num_residual_blocks):
            dilation_rate = 2 ** i
            residual = x
            x = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=2, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
            x = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=1, padding='causal', activation='relu')(x)
            x = tf.keras.layers.add([x, residual])  # Residual connection

        # Final convolutional layer to produce the output
        x = tf.keras.layers.Conv1D(filters=output_shape, kernel_size=1, padding='causal', activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

class ContingentClaim:
    """
    The base class for a financial contingent claim, such as options or other derivatives.

    Arguments:
    - amount (float, optional): The amount of the claim. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Abstract method that must be implemented by subclasses to calculate the payoff.
    """

    def __init__(self, amount=1.0):
        self.amount = amount

    def calculate_payoff(self, paths):
        """
        Abstract method that must be implemented by subclasses to calculate the payoff.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.

        Returns:
        None. (Must be implemented in subclasses.)
        """
        raise NotImplementedError("Subclasses must implement this method.")

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

class CostFunction:
    """
    The base class for a cost function related to financial transactions.

    Methods:
    - calculate(self, actions, paths): Abstract method that must be implemented by subclasses to calculate the cost.
    """

    def calculate(self, actions, paths):
        """
        Abstract method that must be implemented by subclasses to calculate the cost.

        Arguments:
        - actions (tf.Tensor): Tensor containing the actions taken at each time step.
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.

        Returns:
        None. (Must be implemented in subclasses.)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, actions, paths):
        return self.calculate(actions, paths) 

class ProportionalCost(CostFunction):
    """
    A class representing a proportional transaction cost function.

    Arguments:
    - proportion (float): The proportional cost rate, i.e., the cost as a percentage of the transaction.

    Methods:
    - calculate(self, actions, paths): Calculates the proportional transaction cost using the actions and paths.
      - Arguments:
        - actions (tf.Tensor): Tensor containing the actions taken at each time step.
          Expected shape is (num_paths, N).
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
          Expected shape is (num_paths, N+1).
      - Returns:
        - cost (tf.Tensor): Tensor containing the calculated transaction costs.
          Shape is (num_paths, N).
    """

    def __init__(self, proportion):
        self.proportion = proportion

    def calculate(self, actions, paths):
        """
        Calculates the proportional transaction cost using the actions and paths.

        Arguments:
        - actions (tf.Tensor): Tensor containing the actions taken at each time step.
          Expected shape is (num_paths, N).
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
          Expected shape is (num_paths, N+1).

        Returns:
        - cost (tf.Tensor): Tensor containing the calculated transaction costs.
          Shape is (num_paths, N).
        """
        # Calculate the cost as the product of the proportion, the actions, and the corresponding stock prices
        cost = self.proportion * tf.abs(actions) * paths
        
        return cost

class Environment:
    def __init__(self, agent, T, N, r, instrument_list, n_instruments, contingent_claim, cost_function, risk_measure,
                 n_epochs, batch_size, learning_rate, optimizer):
        self.agent = agent
        self.T = T # Maturity (in years)
        self.N = N # Number of hedging steps
        self.dt = self.T / self.N # Time increment
        self.r = r # Risk Free yearly rate
        self.instrument_list = instrument_list
        self.n_instruments = n_instruments # It may not be the same as len of instrument_list (e.g. HestonStock with return_variance)
        self.contingent_claim = contingent_claim
        self.cost_function = cost_function
        self.risk_measure = risk_measure
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def generate_data(self, n_paths, random_seed=None):

        data = tf.TensorArray(dtype=tf.float32, size=self.n_instruments)
        
        i = 0
        for instrument in self.instrument_list:
            instrument_data = instrument.generate_paths(n_paths, random_seed=random_seed)
            if isinstance(instrument_data, tuple):
                for individual_data in instrument_data:
                    data = data.write(i, individual_data)
                    i += 1
            else:
                data = data.write(i, instrument_data)
                i += 1
        
        # Stack all the instrument data into a single tensor
        data = data.stack()

        data_transposed = tf.transpose(data, perm=[1, 2, 0])
        
        return data_transposed # (n_paths, N+1, n_instruments)


    def calculate_pnl(self, paths, actions, include_decomposition = False):
        # Calculate the portfolio value at each time step
        portfolio_values = tf.cumsum(actions, axis=1) * paths # (batch_size, N+1, n_instruments)        
        final_portfolio_value = portfolio_values[:, -1, :] # (batch_size, n_instruments)
        final_portfolio_value = tf.reduce_sum(final_portfolio_value, axis = 1) # (batch_size)

        # Calculate the total transaction costs
        costs = self.cost_function(actions, paths)

        # Calculate the cash flows from purchases and transaction costs
        purchases_cashflows = -actions * paths - costs
        purchases_cashflows = tf.reduce_sum(purchases_cashflows, axis = 2)

        # Calculate the factor for compounding cash positions
        factor = (1 + self.r) ** self.dt - 1

        # Initialize the cash tensor with the first cash flow
        cash = purchases_cashflows[:, 0:1]

        # Iterate over the remaining time steps to accumulate cash values
        for t in range(1, purchases_cashflows.shape[1]):
            current_cash = cash[:, -1:] * (1 + factor) + purchases_cashflows[:, t:t+1]
            cash = tf.concat([cash, current_cash], axis=1)

        final_cash_positions = cash[:, -1] # (batch_size)

        # Calculate the payoff of the contingent claim
        payoff = self.contingent_claim.calculate_payoff(paths[:, :, 0]) # ASSUMPTION: The CC is calculated based on the first instrument
        
        # Calculate the PnL
        pnl = final_portfolio_value + final_cash_positions - payoff
        
        if include_decomposition:
            return pnl, portfolio_values, cash, payoff
        else:
            return pnl

    def loss_function(self, paths, actions):
        pnl = self.calculate_pnl(paths, actions)
        return self.risk_measure.calculate(pnl)

    def train(self, train_paths, val_paths = 0):

        train_data = self.generate_data(train_paths) # (n_paths, N+1, n_instruments)
        T_minus_t = self.get_T_minus_t(self.batch_size) # (n_paths, N)
    
        if val_paths > 0:
            val_data = self.generate_data(val_paths)
            T_minus_t_val =  self.get_T_minus_t(val_paths)

        for epoch in range(self.n_epochs):
            # Training
            epoch_losses = []
            for i in range(0, train_paths, self.batch_size):
                batch_paths = train_data[i:i+self.batch_size]
                loss = self.agent.train_batch(batch_paths, T_minus_t, self.optimizer, self.loss_function)
                epoch_losses.append(loss.numpy())

            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)

            # Validation
            if val_paths > 0:
                val_actions = self.agent.process_batch(val_data, T_minus_t_val)
                val_loss = self.loss_function(val_data, val_actions)
                self.val_losses.append(val_loss.numpy())
                print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.learning_rate}")
            else:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, LR: {self.optimizer.learning_rate}")

        if val_paths > 0:
            return self.train_losses, self.val_losses
        else:
            return self.train_losses

    def test(self, paths_to_test = None, n_paths = None, random_seed = None, 
             plot_pnl = False, plot_title = 'PnL distribution', save_plot_path = None):
        
        if paths_to_test:
            paths = paths_to_test
        elif n_paths:
            paths = self.generate_paths(n_paths, random_seed = random_seed)
        else:
            raise ValueError('Insert either paths_to_test or n_paths.')
        
        T_minus_t_single = tf.range(self.N, 0, -1, dtype=tf.float32) * self.dt
        T_minus_t = tf.tile(tf.expand_dims(T_minus_t_single, axis=0), [paths.shape[0], 1])

        val_actions = self.agent.process_batch(paths, T_minus_t)
        loss = self.loss_function(paths, val_actions)

        if plot_pnl:
            pnl = self.calculate_pnl(paths, val_actions)
            
            # Improved histogram plot
            plt.figure(figsize=(10, 6))
            plt.hist(pnl, bins=30, color='blue', alpha=0.7, edgecolor='black')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'{plot_title}. Loss: {loss:.4f}', fontsize=14)
            plt.xlabel('PnL', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)

            if save_plot_path:
                plt.savefig(save_plot_path)
                print(f"Plot saved to {save_plot_path}")
            else:
                plt.show()
                
        return loss

    def get_T_minus_t(self, shape):
        T_minus_t_single = tf.range(self.N, 0, -1, dtype=tf.float32) * self.dt
        T_minus_t = tf.tile(tf.expand_dims(T_minus_t_single, axis=0), [shape, 1])
        return T_minus_t

    def terminal_hedging_error(self, paths_to_test = None, n_paths = None, random_seed = None, 
                               fixed_price = None, n_paths_for_pricing = None, plot_error = False, 
                               plot_title = 'Terminal Hedging Error', save_plot_path = None):
        
        if paths_to_test:
            paths = paths_to_test
        elif n_paths:
            paths = self.generate_data(n_paths, random_seed = random_seed)
        else:
            raise ValueError('Insert either paths_to_test or n_paths.')
        
        if fixed_price:
            price = fixed_price
        elif n_paths_for_pricing:
            price_paths = self.generate_data(n_paths)
            T_minus_t = self.get_T_minus_t(price_paths.shape[0])
            actions = self.agent.process_batch(paths, T_minus_t)
            loss = self.loss_function(paths, actions)
            price = loss * np.exp(-self.r * self.T)
            print(price)
        else:
            raise ValueError('Insert either fixed_price or n_paths_for_pricing.')
        
        T_minus_t = self.get_T_minus_t(paths.shape[0])
        val_actions = self.agent.process_batch(paths, T_minus_t)
        loss = self.loss_function(paths, val_actions)

        pnl = self.calculate_pnl(paths, val_actions)
        error = price + pnl * np.exp(-self.r * self.T)
        mean_error = tf.reduce_mean(error)        

        if plot_error:
            plt.figure(figsize=(10, 6))
            plt.hist(error, bins=30, color='blue', alpha=0.7, edgecolor='black')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'{plot_title}. Mean Error: {mean_error:.4f}', fontsize=14)
            plt.xlabel('Error', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)

            if save_plot_path:
                plt.savefig(save_plot_path)
                print(f"Plot saved to {save_plot_path}")
            else:
                plt.show()
                
        return mean_error

    def terminal_hedging_error_multiple_agents(self, agents, n_paths=10_000, random_seed=None, 
                                                fixed_price=None, plot_error=False, 
                                                plot_title='Terminal Hedging Error', save_plot_path=None, 
                                                colors=None, save_stats_path=None, loss_functions=None):
        """
        Computes terminal hedging error for multiple agents, generates plots, and saves statistics.

        Arguments:
        - agents (list): List of agent instances to evaluate.
        - n_paths (int): Number of paths to generate for evaluation.
        - random_seed (int or None): Random seed for path generation.
        - fixed_price (float): The fixed price to use in calculating the hedging error.
        - plot_error (bool): Whether to plot the error histograms.
        - plot_title (str): Title of the plot.
        - save_plot_path (str or None): Path to save the plot image. If None, plot is shown.
        - colors (list or None): List of colors for the plot. If None, default colors are used.
        - save_stats_path (str or None): Path to save the statistics as an Excel file.
        - loss_functions (list of callables or None): List of loss functions to evaluate on the PnL.

        Returns:
        - mean_errors (list): List of mean errors for each agent.
        - std_errors (list): List of standard deviations of errors for each agent.
        - losses (list of lists): List of loss values for each agent and loss function.
        """
        
        paths = self.generate_data(n_paths, random_seed=random_seed)

        if fixed_price is not None:
            price = fixed_price
        else:
            raise ValueError('Insert fixed_price.')

        errors = []
        mean_errors = []
        std_errors = []

        # Initialize a dictionary to store losses for each loss function and agent
        loss_results = {loss_fn.name: [] for loss_fn in loss_functions} if loss_functions else {}

        for agent in agents:
            T_minus_t = self.get_T_minus_t(paths.shape[0])
            val_actions = agent.process_batch(paths, T_minus_t)
            pnl = self.calculate_pnl(paths, val_actions)
            error = price + pnl * np.exp(-self.r * self.T)
            
            errors.append(error)
            mean_errors.append(tf.reduce_mean(error).numpy())
            std_errors.append(tf.math.reduce_std(error).numpy())

            # Compute additional loss functions if provided
            if loss_functions:
                for loss_fn in loss_functions:
                    loss_value = loss_fn(pnl)
                    loss_results[loss_fn.name].append(tf.reduce_mean(loss_value).numpy())

        if plot_error:
            plt.figure(figsize=(10, 6))

            # Calculate combined bin edges to ensure consistent bins across all histograms
            min_error = -19.  # Fixed minimum error range for bins
            max_error = 4.   # Fixed maximum error range for bins
            bins = np.linspace(min_error, max_error, 60)  # 60 bins across the range of all errors

            if colors is None:
                cmap = plt.cm.get_cmap('tab10', len(agents))
                colors = [cmap(i) for i in range(len(agents))]

            for i, error in enumerate(errors):
                plt.hist(error, bins=bins, color=colors[i], alpha=0.8, edgecolor='black', 
                        label=f'{agents[i].plot_name} Agent')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'{plot_title}', fontsize=14)
            plt.xlabel('Error', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()

            if save_plot_path:
                plt.savefig(save_plot_path)
                print(f"Plot saved to {save_plot_path}")
            else:
                plt.show()

        # Save statistics to Excel if save_stats_path is provided
        if save_stats_path:
            data = {
                'Agent': [agent.plot_name for agent in agents],
                'Mean Error': mean_errors,
                'Standard Deviation': std_errors,
            }
            # Add additional loss functions to the data dictionary
            if loss_functions:
                for loss_fn_name, results in loss_results.items():
                    data[loss_fn_name] = results

            df = pd.DataFrame(data)
            df.to_excel(save_stats_path, index=False)
            print(f"Statistics saved to {save_stats_path}")
            return df

        return mean_errors, std_errors, loss_results

    def save_optimizer(self, optimizer_path):
        """
        Save the optimizer state and object to the specified path using optimizer.get_config() and optimizer.variables().

        Arguments:
        - optimizer_path (str): Directory path where the optimizer state will be saved.
        """
        # Ensure the directory exists
        os.makedirs(optimizer_path, exist_ok=True)

        # Save the optimizer configuration
        optimizer_config = self.optimizer.get_config()
        with open(os.path.join(optimizer_path, 'optimizer_config.pkl'), 'wb') as f:
            pickle.dump(optimizer_config, f)

        # Save the optimizer variables (weights)
        optimizer_weights = [variable.numpy() for variable in self.optimizer.variables]
        with open(os.path.join(optimizer_path, 'optimizer_weights.pkl'), 'wb') as f:
            pickle.dump(optimizer_weights, f)

    def load_optimizer(self, optimizer_path, only_weights = False):
        """
        Load the optimizer state and object from the specified path using optimizer.get_config() and optimizer.variables().

        Arguments:
        - optimizer_path (str): Directory path from which the optimizer state will be loaded.
        """
        if not os.path.exists(optimizer_path):
            warnings.warn(f"Optimizer path '{optimizer_path}' does not exist. Exiting the load function.")
            return

        variables_index_start = 2
        if not only_weights:
            # Load the optimizer configuration
            with open(os.path.join(optimizer_path, 'optimizer_config.pkl'), 'rb') as f:
                optimizer_config = pickle.load(f)
            
            # Re-instantiate the optimizer using the loaded configuration
            self.optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config)
            variables_index_start = 0

        # Initialize the optimizer's variables by applying it to some dummy data
        dummy_data = [tf.zeros_like(var) for var in self.agent.model.trainable_variables]
        self.optimizer.apply_gradients(zip(dummy_data, self.agent.model.trainable_variables))

        # Load the optimizer weights
        with open(os.path.join(optimizer_path, 'optimizer_weights.pkl'), 'rb') as f:
            optimizer_weights = pickle.load(f)

        # Apply the loaded weights to the optimizer
        for variable, weight in zip(self.optimizer.variables[variables_index_start:], optimizer_weights[variables_index_start:]):
            variable.assign(weight)

    def plot_hedging_strategy(self, save_plot_path=None, random_seed = None):
        """
        Generates one path, gets the hedging actions, and plots the portfolio value over time.
        Also plots the contingent claim payoff at the final timestep and prints the loss value.
        Additionally, plots the underlying asset path on a secondary axis.

        Arguments:
        - save_plot_path (str, optional): File path to save the plot. If None, the plot is only shown.
        """

        # Generate a single path
        single_path = self.generate_paths(1, random_seed = random_seed)  # Shape: (1, N+1)
        T_minus_t_single = tf.range(self.N, 0, -1, dtype=tf.float32) * self.dt
        T_minus_t_single = tf.expand_dims(T_minus_t_single, axis=0)  # Shape: (1, N)

        # Get the hedging actions
        actions = self.agent.process_batch(single_path, T_minus_t_single)  # Shape: (1, N)

        pnl, portfolio_values, cash, payoff = self.calculate_pnl(single_path, actions, include_decomposition = True)

        net_portfolio_values = portfolio_values + cash
        net_portfolio_values = net_portfolio_values.numpy()[0]

        # Plot the portfolio value over time
        timesteps = np.arange(self.N + 1)
        fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
        ax1.bar(timesteps, net_portfolio_values, label="Net Portfolio Values", color='b', alpha=0.6)

        # Plot the contingent claim payoff at the final timestep as a bar
        ax1.bar(self.N, payoff, color='r', alpha=0.6, label="Contingent Claim Payoff")

        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Portfolio Value", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a secondary y-axis for the underlying asset price
        ax2 = ax1.twinx()
        ax2.plot(timesteps, single_path.numpy().flatten(), label="Underlying Asset Path", color='g', alpha=0.6)
        ax2.set_ylabel("Underlying Asset Price", color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.axhline(y=self.contingent_claim.strike, color='orange', linestyle='-', label="Strike Price")

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # Title and layout
        plt.title(f"Hedging Strategy Over Time. PnL: {pnl.numpy()[0]:.4f}")
        fig.tight_layout()

        # Save the plot if a path is provided, otherwise show it
        if save_plot_path:
            plt.savefig(save_plot_path)
            print(f"Plot saved to {save_plot_path}")
        else:
            plt.show()

class Stock:
    """
    The base class for simulating stock price paths. This class provides the basic
    structure and methods that must be implemented by subclasses.

    Arguments:
    - S0 (float): Initial stock price.
    - T (float): Time horizon for the simulation.
    - N (int): Number of time steps in the simulation.
    - r (float): Risk-free interest rate.

    Methods:
    - generate_paths(self): Abstract method that must be implemented by subclasses to generate stock price paths.
    - plot(self, paths, title="Stock Price Paths"): Plots the stock price paths.
    """
    def __init__(self, S0, T, N, r):
        self.S0 = S0  # Initial stock price
        self.T = T    # Time horizon
        self.N = N    # Number of time steps
        self.r = r    # Risk-free rate
        self.dt = T / N  # Time increment

    def generate_paths(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def plot(self, paths, title="Stock Price Paths"):
        plt.figure(figsize=(10, 6))
        plt.plot(paths.numpy().T)
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.show()

class GBMStock(Stock):
    """
    A subclass of Stock that models stock prices using the Geometric Brownian Motion (GBM) model.

    Arguments:
    - S0 (float): Initial stock price.
    - T (float): Time horizon for the simulation.
    - N (int): Number of time steps in the simulation.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the stock.

    Methods:
    - generate_paths(self, num_paths): Generates stock price paths using the GBM model.
    """
    def __init__(self, S0, T, N, r, sigma):
        super().__init__(S0, T, N, r)
        self.sigma = sigma  # Stock volatility

    def generate_paths(self, num_paths, random_seed = None):
        """
        Generates stock price paths using the GBM model.

        Arguments:
        - num_paths (int): Number of paths to simulate.
        - random_seed (int, optional): Seed for random number generation. Default is None.

        Returns:
        - S_paths (tf.Tensor): A TensorFlow tensor containing the generated stock price paths.
          Shape is (num_paths, N+1).
        """
        dt = self.dt
        S0 = self.S0
        r = self.r
        sigma = self.sigma

        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random normal variables for the Brownian motion
        dW = np.random.normal(0, 1, size=(num_paths, self.N)) * np.sqrt(dt)
        
        # Initialize the matrix of paths
        S = np.zeros((num_paths, self.N + 1))
        S[:, 0] = S0
        
        # Generate the paths using the GBM formula
        for t in range(1, self.N + 1):
            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW[:, t-1])
        
        # Convert the paths to a TensorFlow tensor
        S_paths = tf.convert_to_tensor(S, dtype=tf.float32)
        
        return S_paths

class HestonStock(Stock):
    """
    A subclass of Stock that models stock prices using the Heston stochastic volatility model.

    Arguments:
    - S0 (float): Initial stock price.
    - T (float): Time horizon for the simulation.
    - N (int): Number of time steps in the simulation.
    - r (float): Risk-free interest rate.
    - v0 (float): Initial variance.
    - kappa (float): Rate of mean reversion for the variance.
    - theta (float): Long-term variance (mean level).
    - xi (float): Volatility of variance (volatility of volatility).
    - rho (float): Correlation between the Brownian motions driving the stock price and variance.
    - return_variance(bool): True
    Methods:
    - generate_paths(self, num_paths): Generates stock price paths using the Heston model,
      with an option to return the variance paths.
    """
    def __init__(self, S0, T, N, r, v0, kappa, theta, xi, rho, return_variance=True):
        super().__init__(S0, T, N, r)
        self.v0 = v0        # Initial variance
        self.kappa = kappa  # Rate of reversion
        self.theta = theta  # Long-term variance
        self.xi = xi        # Volatility of variance
        self.rho = rho      # Correlation between Brownian motions
        self.return_variance = return_variance

    def generate_paths(self, num_paths, random_seed = None):
        """
        Generates stock price paths using the Heston model, with an option to return the variance paths.

        Arguments:
        - num_paths (int): Number of paths to simulate.
        - return_variance (bool, optional): If True, the method returns both the stock price paths and the variance paths.
          Default is False.
        - random_seed (int, optional): Seed for random number generation. Default is None.

        Returns:
        - S_paths (tf.Tensor): A TensorFlow tensor containing the generated stock price paths. Shape is (num_paths, N+1).
        - v_paths (tf.Tensor, optional): A TensorFlow tensor containing the generated variance paths,
          returned if return_variance is True. Shape is (num_paths, N+1).
        """
        dt = self.dt
        S0 = self.S0
        r = self.r
        v0 = self.v0
        kappa = self.kappa
        theta = self.theta
        xi = self.xi
        rho = self.rho

        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate correlated random normal variables
        dW1 = np.random.normal(0, 1, size=(num_paths, self.N)) * np.sqrt(dt)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, size=(num_paths, self.N)) * np.sqrt(dt)

        # Initialize the paths for stock prices and variances
        S = np.zeros((num_paths, self.N + 1))
        v = np.zeros((num_paths, self.N + 1))
        S[:, 0] = S0
        v[:, 0] = v0

        # Generate the paths using the Heston model
        for t in range(1, self.N + 1):
            v[:, t] = v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + xi * np.sqrt(v[:, t-1]) * dW2[:, t-1]
            v[:, t] = np.maximum(v[:, t], 0)  # Ensure variance stays non-negative

            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t]) * dW1[:, t-1])

        # Convert the paths to TensorFlow tensors
        S_paths = tf.convert_to_tensor(S, dtype=tf.float32)
        v_paths = tf.convert_to_tensor(v, dtype=tf.float32)

        if self.return_variance:
            return S_paths, v_paths
        else:
            return S_paths

class RiskMeasure:
    """
    The base class for risk measures that take a PnL (Profit and Loss) tensor and return a corresponding risk measure.

    Methods:
    - calculate(self, pnl): Abstract method that must be implemented by subclasses to calculate the risk measure.
    """
    def __init__(self):
        self.name = 'unknown'

    def calculate(self, pnl):
        """
        Abstract method that must be implemented by subclasses to calculate the risk measure.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        None. (Must be implemented in subclasses.)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, pnl):
        return self.calculate(pnl)
    
class MSE(RiskMeasure):
    """
    A class representing the Mean Squared Error (MSE) risk measure.

    Methods:
    - calculate(self, pnl): Calculates the MSE of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - mse (tf.Tensor): Scalar tensor containing the calculated MSE.
    """
    def __init__(self):
        self.name = 'MSE'

    def calculate(self, pnl):
        """
        Calculates the Mean Squared Error (MSE) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - mse (tf.Tensor): Scalar tensor containing the calculated MSE.
        """
        mse = tf.reduce_mean(tf.square(pnl))
        return mse

class SMSE(RiskMeasure):
    """
    A class representing the Semi Mean Squared Error (SMSE) risk measure,
    which penalizes only negative PnL.

    Methods:
    - calculate(self, pnl): Calculates the SMSE of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - smse (tf.Tensor): Scalar tensor containing the calculated SMSE.
    """
    def __init__(self):
        self.name = 'SMSE'

    def calculate(self, pnl):
        """
        Calculates the Semi Mean Squared Error (SMSE) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - smse (tf.Tensor): Scalar tensor containing the calculated SMSE.
        """
        negative_pnl = tf.boolean_mask(pnl, pnl < 0)
        smse = tf.reduce_mean(tf.square(negative_pnl))
        return smse

class MAE(RiskMeasure):
    """
    A class representing the Mean Absolute Error (MAE) risk measure.

    Methods:
    - calculate(self, pnl): Calculates the MAE of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - mae (tf.Tensor): Scalar tensor containing the calculated MAE.
    """
    def __init__(self):
        self.name = 'MAE'

    def calculate(self, pnl):
        """
        Calculates the Mean Absolute Error (MAE) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - mae (tf.Tensor): Scalar tensor containing the calculated MAE.
        """
        mae = tf.reduce_mean(tf.abs(pnl))
        return mae

class VaR(RiskMeasure):
    """
    A class representing the Value at Risk (VaR) risk measure.

    Arguments:
    - alpha (float): The confidence level (e.g., 0.95 for 95% confidence).

    Methods:
    - calculate(self, pnl): Calculates the VaR of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - var (tf.Tensor): Scalar tensor containing the calculated VaR.
    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.name = f'VaR_{int(alpha*100)}'

    def calculate(self, pnl):
        """
        Calculates the Value at Risk (VaR) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - var (tf.Tensor): Scalar tensor containing the calculated VaR.
        """
        var = tfp.stats.percentile(pnl, q=(1 - self.alpha) * 100, interpolation='linear')
        return -var

class CVaR(RiskMeasure):
    """
    A class representing the Conditional Value at Risk (CVaR) risk measure.

    Arguments:
    - alpha (float): The confidence level (e.g., 0.95 for 95% confidence).

    Methods:
    - calculate(self, pnl): Calculates the CVaR of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - cvar (tf.Tensor): Scalar tensor containing the calculated CVaR.
    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.name = f'CVaR_{int(alpha*100)}'

    def calculate(self, pnl):
        """
        Calculates the Conditional Value at Risk (CVaR) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - cvar (tf.Tensor): Scalar tensor containing the calculated CVaR.
        """
        # Compute the Value at Risk (VaR)
        var = tfp.stats.percentile(pnl, q=(1 - self.alpha) * 100, interpolation='linear')

        # Compute CVaR by taking the mean of the losses that are less than or equal to VaR
        cvar = tf.reduce_mean(tf.boolean_mask(pnl, pnl <= var))

        return -cvar

class WorstCase(RiskMeasure):
    """
    A class representing the Worst Case risk measure, 
    which returns the minimum value of the PnL (worst-case scenario).

    Methods:
    - calculate(self, pnl): Calculates the worst-case scenario of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - worst_case (tf.Tensor): Scalar tensor containing the calculated worst-case scenario.
    """
    def __init__(self):
        self.name = 'WorstCase'

    def calculate(self, pnl):
        """
        Calculates the Worst Case scenario of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - worst_case (tf.Tensor): Scalar tensor containing the calculated worst-case scenario.
        """
        worst_case = tf.reduce_min(pnl)
        return -worst_case

class Entropy(RiskMeasure):
    """
    A class representing the Entropy risk measure, 
    which quantifies the uncertainty in the PnL distribution.

    Methods:
    - calculate(self, pnl): Calculates the entropy of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - entropy (tf.Tensor): Scalar tensor containing the calculated entropy.
    """
    def __init__(self):
        self.name = 'Entropy'

    def calculate(self, pnl):
        """
        Calculates the entropy of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - entropy (tf.Tensor): Scalar tensor containing the calculated entropy.
        """
        # Normalize the PnL values to obtain a probability distribution
        normalized_pnl = pnl - tf.reduce_min(pnl) + 1e-9  # Shift to positive values to avoid log(0)
        probabilities = normalized_pnl / tf.reduce_sum(normalized_pnl)

        # Compute the entropy
        entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities))
        return entropy

T = 63/252
t_choice = 31/252
N = 63
r = 0.05
n_instruments = 1

instrument1 = GBMStock(S0=100, T=T, N=N, r=r, sigma=0.2)

instruments = [instrument1] #instrument1
contingent_claim = ChooserOption(strike=100,
                                 t_choice=t_choice,
                                 r=instrument1.r,
                                 sigma=instrument1.sigma,
                                 T=instrument1.T)

path_transformation_configs = [
    {'transformation_type': 'log_moneyness', 'K': contingent_claim.strike}
]

cost_function = ProportionalCost(proportion=0.0)
risk_measure = CVaR(alpha=0.5)

agent = RecurrentAgent(path_transformation_configs=path_transformation_configs)
#agent = LSTMAgent(N, path_transformation_configs=path_transformation_configs)
#agent = GRUAgent(N, path_transformation_configs=path_transformation_configs)
#agent = WaveNetAgent(N, path_transformation_configs=path_transformation_configs)

model_name = 'chooser_1'
model_path = os.path.join(os.getcwd(), 'models', agent.name, f'{model_name}.keras')
optimizer_path = os.path.join(os.getcwd(), 'optimizers', agent.name, model_name)

print(agent.name, model_name)
initial_learning_rate = 0.001
decay_steps = 10 
decay_rate = 0.99  

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True 
)

env = Environment(
    agent=agent,
    T = T,
    N = N,
    r = r,
    instrument_list=instruments,
    n_instruments = n_instruments,
    contingent_claim=contingent_claim,
    cost_function=cost_function,
    risk_measure=risk_measure,
    n_epochs=100,
    batch_size=10_000,
    learning_rate=learning_rate_schedule,
    optimizer=tf.keras.optimizers.Adam
)

print(time.ctime())

agent.load_model(model_path)
env.load_optimizer(optimizer_path, only_weights=True)

env.train(train_paths=100_000)

agent.save_model(model_path)
env.save_optimizer(optimizer_path)

print(time.ctime())

