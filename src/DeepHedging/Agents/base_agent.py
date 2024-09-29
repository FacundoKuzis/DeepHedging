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

    def transform_input(self, instrument_paths, T_minus_t, additional_info=None):
        """
        Transforms the input by concatenating instrument paths, time to maturity, and additional information.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.
        - additional_info (tf.Tensor, optional): Additional information not related to tradable assets.

        Returns:
        - input_data (tf.Tensor): The transformed input data.
        """
        instrument_paths = self.transform_paths(instrument_paths, self.path_transformation_configs)
        T_minus_t_expanded = tf.expand_dims(T_minus_t, axis=-1)
        input_data = tf.concat([instrument_paths, T_minus_t_expanded], axis=-1)
        if additional_info is not None:
            # Ensure additional_info has the correct shape
            if len(additional_info.shape) == 2:
                additional_info_expanded = tf.expand_dims(additional_info, axis=1)
                # Tile along time dimension
                additional_info_expanded = tf.tile(additional_info_expanded, [1, input_data.shape[1], 1])
            else:
                additional_info_expanded = additional_info
            input_data = tf.concat([input_data, additional_info_expanded], axis=-1)
        return input_data

    def act(self, instrument_paths, T_minus_t, additional_info=None):
        """
        Act based on the input.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.
        - additional_info (tf.Tensor, optional): Additional information not related to tradable assets.

        Returns:
        - action (tf.Tensor): The action chosen by the model.
        """
        input_data = self.transform_input(instrument_paths, T_minus_t, additional_info)
        action = self.model(input_data)
        return action

    def train_batch(self, batch_paths, batch_T_minus_t, optimizer, loss_function, additional_info=None):
        with tf.GradientTape() as tape:
            actions = self.process_batch(batch_paths, batch_T_minus_t, additional_info)
            loss = loss_function(batch_paths, actions)
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