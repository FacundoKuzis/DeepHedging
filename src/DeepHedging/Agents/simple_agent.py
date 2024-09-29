import tensorflow as tf
from DeepHedging.Agents import BaseAgent

class SimpleAgent(BaseAgent):
    """
    A simple agent that processes inputs timestep by timestep.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    def __init__(self, path_transformation_configs = None, n_instruments = 1,
                 n_additional_features=0):
        self.input_shape = (n_instruments + 1 + n_additional_features,) # +1 for T-t
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

    def process_batch(self, batch_paths, batch_T_minus_t, additional_information = None):
        all_actions = []
        for t in range(batch_paths.shape[1] -1):  # timesteps until T-1
            current_paths = batch_paths[:, t, :] # (n_simulations, n_timesteps, n_instruments)
            current_T_minus_t = batch_T_minus_t[:, t] # (n_simulations, n_timesteps)
            action = self.act(current_paths, current_T_minus_t, additional_information)
            all_actions.append(action)

        all_actions = tf.stack(all_actions, axis=1)
        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[-1]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)

        return all_actions # (n_simulations, n_timesteps, n_instruments)
