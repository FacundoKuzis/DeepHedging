import tensorflow as tf
from abc import ABC, abstractmethod

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
        return args  # By default, return inputs as-is
