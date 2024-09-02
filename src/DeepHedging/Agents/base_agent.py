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