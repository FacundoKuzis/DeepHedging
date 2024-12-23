import tensorflow as tf
from DeepHedging.Agents import BaseAgent

class LSTMAgent(BaseAgent):
    """
    An LSTM agent that processes the entire sequence of inputs at once.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    plot_color = 'forestgreen' 
    name = 'lstm'
    is_trainable = True
    plot_name = {
        'en': 'LSTM Agent',
        'es': 'Agente LSTM'
    }

    def __init__(self, n_hedging_timesteps, path_transformation_configs = None,
                 n_instruments = 1):

        self.input_shape = (n_hedging_timesteps, n_instruments + 1) # +1 for T-t
        self.n_instruments = n_instruments
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.path_transformation_configs = path_transformation_configs

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

