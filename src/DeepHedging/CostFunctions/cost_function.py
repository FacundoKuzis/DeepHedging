class CostFunction:
    """
    The base class for a cost function related to financial transactions.

    Methods:
    - calculate(self, actions, paths): Abstract method that must be implemented by subclasses to calculate the cost.
    """

    def calculate(self, actions, paths):
        """
        Abstract method that must be implemented by subclasses to calculate the cost.

        Arguments:
        - actions (tf.Tensor): Tensor containing the actions taken at each time step.
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.

        Returns:
        None. (Must be implemented in subclasses.)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, actions, paths):
        return self.calculate(actions, paths) 