import tensorflow as tf
from DeepHedging.Agents import SimpleAgent

class RecurrentAgent(SimpleAgent):
    """
    A recurrent agent that processes inputs timestep by timestep and includes the accumulated position.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """
    plot_color = 'steelblue' 
    name = 'recurrent'
    is_trainable = True
    plot_name = {
        'en': 'Recurrent Agent',
        'es': 'Agente Recurrente'
    }

    def __init__(
        self,
        path_transformation_configs = None,
        n_instruments = 1,
        n_hedging_timesteps = None,
        history_feature_dim = 0,
    ):
        self.history_feature_dim = int(history_feature_dim)
        self.input_shape = (n_instruments + 1 + n_instruments + self.history_feature_dim,) # +1 for T-t and + n_instruments for accumulated position
        self.n_instruments = n_instruments
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.accumulated_position = None  # Initialize accumulated position
        self.path_transformation_configs = path_transformation_configs

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

    def transform_input(self, instrument_paths, T_minus_t, history_features=None):
        """
        Transforms the input by including the accumulated position.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - transformed_input (tf.Tensor): The transformed input, including accumulated position and T_minus_t.
        """
        # Concatenate the instrument paths, accumulated position, and T_minus_t
        input_data = super().transform_input(
            instrument_paths,
            T_minus_t,
            history_features=history_features,
        )
        transformed_input = tf.concat([input_data, self.accumulated_position], axis=-1)

        return transformed_input # (batch_size, n_instruments + 1 + n_instruments)


    def act(self, instrument_paths, T_minus_t, history_features=None):
        """
        Act based on the input.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The action chosen by the model.
        """
        input_data = self.transform_input(
            instrument_paths,
            T_minus_t,
            history_features=history_features,
        )
        action = self.model(input_data)

        # Update accumulated position by summing actions
        self.accumulated_position += action  # Accumulate positions across timesteps

        return action

    def process_batch(self, batch_paths, batch_T_minus_t, batch_history_features=None):
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
        return super().process_batch(
            batch_paths,
            batch_T_minus_t,
            batch_history_features=batch_history_features,
        )
