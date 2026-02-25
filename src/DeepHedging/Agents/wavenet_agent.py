import tensorflow as tf
from DeepHedging.Agents import LSTMAgent

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

    plot_color = 'pink' 
    name = 'wavenet'
    is_trainable = True
    plot_name = {
        'en': 'WaveNet Agent',
        'es': 'Agente WaveNet'
    }
    
    def __init__(
        self,
        n_hedging_timesteps,
        path_transformation_configs=None,
        num_filters=32,
        num_residual_blocks=3,
        n_instruments=1,
        history_feature_dim=0,
        context_as_timesteps=True,
        context_pre_ttm_mode="calculated",
        sequence_output_mode="trade",
        position_activation="linear",
    ):
        
        self.history_feature_dim = int(history_feature_dim)
        # Keep time dimension dynamic so context can be consumed as sequence prefix.
        self.input_shape = (None, n_instruments + 1 + self.history_feature_dim) # +1 for T-t
        self.n_instruments = n_instruments
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.context_as_timesteps = bool(context_as_timesteps)
        out_mode = str(sequence_output_mode).strip().lower()
        if out_mode not in {"trade", "position"}:
            raise ValueError("sequence_output_mode must be 'trade' or 'position'.")
        self.sequence_output_mode = out_mode
        pos_act = str(position_activation).strip().lower()
        if pos_act not in {"linear", "sigmoid", "tanh"}:
            raise ValueError("position_activation must be one of {'linear','sigmoid','tanh'}.")
        self.position_activation = pos_act
        mode = str(context_pre_ttm_mode).strip().lower()
        if mode == "extended":
            mode = "calculated"
        if mode not in {"calculated", "zero"}:
            raise ValueError("context_pre_ttm_mode must be 'calculated' or 'zero'.")
        self.context_pre_ttm_mode = mode
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.path_transformation_configs = path_transformation_configs

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
