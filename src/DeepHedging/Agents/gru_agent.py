import tensorflow as tf
from DeepHedging.Agents import LSTMAgent

class GRUAgent(LSTMAgent):
    """
    A GRU agent that processes the entire sequence of inputs at once.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    plot_color = 'darkcyan' 
    name = 'gru'
    is_trainable = True
    plot_name = {
        'en': 'GRU Agent',
        'es': 'Agente GRU'
    }

    def __init__(
        self,
        n_hedging_timesteps,
        path_transformation_configs=None,
        n_instruments=1,
        history_feature_dim=0,
        context_as_timesteps=True,
        context_pre_ttm_mode="calculated",
    ):
        
        super().__init__(
            n_hedging_timesteps=n_hedging_timesteps,
            path_transformation_configs=path_transformation_configs,
            n_instruments=n_instruments,
            history_feature_dim=history_feature_dim,
            context_as_timesteps=context_as_timesteps,
            context_pre_ttm_mode=context_pre_ttm_mode,
        )

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
