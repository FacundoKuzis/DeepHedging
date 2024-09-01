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

    def __init__(self, n_hedging_timesteps, path_transformation_type=None, K=None, num_filters=32, num_residual_blocks=3):
        input_shape = (n_hedging_timesteps, 2)
        output_shape = 1
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.model = self.build_model(input_shape, output_shape)
        self.path_transformation_type = path_transformation_type
        self.K = K
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
