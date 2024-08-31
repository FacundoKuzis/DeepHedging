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

    @abstractmethod
    def act(self, instrument_paths, T_minus_t):
        pass

    def transform_input(self, *args):
        return args 

    def transform_paths(self, instrument_paths, transformation_type=None, K=None):
        """
        Transforms the instrument paths based on the specified transformation type.

        Arguments:
        - instrument_paths (tf.Tensor): The paths of the instrument.
        - transformation_type (str, optional): The type of transformation ('log', 'log_moneyness'). Default is None.
        - K (float, optional): The strike price, required if transformation_type is 'log_moneyness'.

        Returns:
        - transformed_paths (tf.Tensor): The transformed instrument paths.
        """

        if transformation_type is None:
            # No transformation, return the paths as-is
            return instrument_paths

        elif transformation_type == 'log':
            # Apply log transformation
            return tf.math.log(instrument_paths)

        elif transformation_type == 'log_moneyness':
            if K is None:
                raise ValueError("Strike price K must be provided for 'log moneyness' transformation.")
            return tf.math.log(instrument_paths / K)

        else:
            raise ValueError(f"Unsupported transformation type: {transformation_type}")

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