import tensorflow as tf


class ContingentClaim:
    """
    The base class for a financial contingent claim, such as options or other derivatives.

    Arguments:
    - amount (float, optional): The amount of the claim. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Abstract method that must be implemented by subclasses to calculate the payoff.
    """

    def __init__(self, amount=1.0, underlying_index=0):
        self.amount = amount
        self.underlying_index = underlying_index

    def select_underlying_paths(self, paths):
        """
        Selects the underlying path(s) used by the claim payoff.

        Supports:
        - int: a single instrument index (returns shape: batch x time)
        - list/tuple/tf.Tensor of ints: multiple instruments (returns shape: batch x time x k)
        """
        if len(paths.shape) == 2:
            return paths

        if len(paths.shape) != 3:
            raise ValueError(
                f"Expected paths rank 2 or 3, got rank {len(paths.shape)}."
            )

        idx = self.underlying_index
        if isinstance(idx, int):
            return paths[:, :, idx]

        if isinstance(idx, (list, tuple)):
            if not idx:
                raise ValueError("underlying_index list cannot be empty.")
            return tf.gather(paths, indices=list(idx), axis=2)

        if isinstance(idx, tf.Tensor):
            return tf.gather(paths, indices=idx, axis=2)

        raise TypeError(
            "underlying_index must be int, list, tuple, or tf.Tensor of indices."
        )

    def calculate_payoff(self, paths):
        """
        Abstract method that must be implemented by subclasses to calculate the payoff.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.

        Returns:
        None. (Must be implemented in subclasses.)
        """
        raise NotImplementedError("Subclasses must implement this method.")
