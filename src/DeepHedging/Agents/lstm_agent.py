import tensorflow as tf
from DeepHedging.Agents import BaseAgent

class LSTMAgent(BaseAgent):
    """
    An LSTM agent that processes the entire sequence of inputs at once.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    def __init__(self, n_hedging_timesteps, path_transformation_type = None, K = None):
        input_shape = (n_hedging_timesteps, 2)
        output_shape = 1
        self.model = self.build_model(input_shape, output_shape)
        self.path_transformation_type = path_transformation_type
        self.K = K
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
            tf.keras.layers.LSTM(50, return_sequences=True, activation = 'relu'),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dense(output_shape, activation='linear')
        ])
        return model


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
        instrument_paths = self.transform_paths(instrument_paths, self.path_transformation_type, K = self.K)

        # Ensure both tensors have the same rank by expanding dimensions of instrument_paths
        instrument_paths = tf.expand_dims(instrument_paths, axis=-1)  # Shape: (batch_size, input_shape, 1)
        T_minus_t = tf.expand_dims(T_minus_t, axis=-1)  # Shape: (batch_size, 1)
        
        # Concatenate along the last axis
        input_data = tf.concat([instrument_paths, T_minus_t], axis=-1)  # Shape: (batch_size, input_shape + 1)
        return input_data
    



    def act(self, instrument_paths, T_minus_t):
        """
        Act based on the entire sequence of inputs.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths for all timesteps.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity for all timesteps.

        Returns:
        - actions (tf.Tensor): The actions chosen by the model.
        """
        input_data = self.transform_input(instrument_paths, T_minus_t)
        actions = self.model(input_data)
        return actions

    def process_batch(self, batch_paths, batch_T_minus_t):

        all_actions = self.act(batch_paths[:, :-1], batch_T_minus_t)

        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[2]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)
        all_actions = all_actions[:, :, 0] # temporary, only one instrument

        return all_actions


    def train_batch(self, batch_paths, batch_T_minus_t, optimizer, loss_function):
        """
        Train the model on a batch of data using the entire sequence at once.

        Arguments:
        - batch_paths (tf.Tensor): Tensor containing a batch of instrument paths. Shape: (batch_size, timesteps, input_shape)
        - batch_T_minus_t (tf.Tensor): Tensor containing the time to maturity at each timestep.
        - optimizer (tf.optimizers.Optimizer): The optimizer to use for training.

        Returns:
        - loss (tf.Tensor): The loss value after training on the batch.
        """
        with tf.GradientTape() as tape:
            actions = self.process_batch(batch_paths, batch_T_minus_t)
            loss = loss_function(batch_paths, actions) 
            # Compute gradients based on the loss
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss
    
