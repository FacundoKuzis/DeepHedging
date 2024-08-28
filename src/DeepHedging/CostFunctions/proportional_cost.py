import tensorflow as tf
from DeepHedging.CostFunctions import CostFunction

class ProportionalCost(CostFunction):
    """
    A class representing a proportional transaction cost function.

    Arguments:
    - proportion (float): The proportional cost rate, i.e., the cost as a percentage of the transaction.

    Methods:
    - calculate(self, actions, paths): Calculates the proportional transaction cost using the actions and paths.
      - Arguments:
        - actions (tf.Tensor): Tensor containing the actions taken at each time step.
          Expected shape is (num_paths, N).
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
          Expected shape is (num_paths, N+1).
      - Returns:
        - cost (tf.Tensor): Tensor containing the calculated transaction costs.
          Shape is (num_paths, N).
    """

    def __init__(self, proportion):
        self.proportion = proportion

    def calculate(self, actions, paths):
        """
        Calculates the proportional transaction cost using the actions and paths.

        Arguments:
        - actions (tf.Tensor): Tensor containing the actions taken at each time step.
          Expected shape is (num_paths, N).
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
          Expected shape is (num_paths, N+1).

        Returns:
        - cost (tf.Tensor): Tensor containing the calculated transaction costs.
          Shape is (num_paths, N).
        """
        # Calculate the cost as the product of the proportion, the actions, and the corresponding stock prices
        cost = self.proportion * tf.abs(actions) * paths
        
        return cost
