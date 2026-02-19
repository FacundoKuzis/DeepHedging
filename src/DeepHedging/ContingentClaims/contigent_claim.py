import tensorflow as tf
import numpy as np


class ContingentClaim:
    """
    The base class for a financial contingent claim, such as options or other derivatives.

    Arguments:
    - amount (float, optional): The amount of the claim. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Abstract method that must be implemented by subclasses to calculate the payoff.
    """

    def __init__(self, amount=1.0, underlying_index=0, fixing_indices=None):
        self.amount = amount
        self.underlying_index = underlying_index
        self.fixing_indices = fixing_indices

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

    def resolve_fixing_indices(self, n_steps):
        """
        Resolve the fixing calendar against a path length.
        Returns a sorted np.ndarray of unique integer indices in [0, n_steps-1].
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        if self.fixing_indices is None:
            return np.arange(n_steps, dtype=np.int32)

        fixings = np.array(self.fixing_indices, dtype=np.int32).reshape(-1)
        if fixings.size == 0:
            raise ValueError("fixing_indices cannot be empty.")
        if np.any(fixings < 0) or np.any(fixings >= n_steps):
            raise ValueError(
                f"fixing_indices must be within [0, {n_steps - 1}]. Got {fixings}."
            )
        fixings = np.unique(fixings)
        fixings.sort()
        return fixings

    def select_fixing_paths(self, paths):
        """
        Select only fixing timestamps from 2D (batch,time) or 3D (batch,time,features) paths.
        """
        n_steps = int(paths.shape[1])
        fixings = self.resolve_fixing_indices(n_steps)
        return tf.gather(paths, indices=fixings, axis=1)
