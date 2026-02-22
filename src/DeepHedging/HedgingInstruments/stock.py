import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Stock:
    """
    The base class for simulating stock price paths. This class provides the basic
    structure and methods that must be implemented by subclasses.

    Arguments:
    - S0 (float): Initial stock price.
    - T (float): Time horizon for the simulation.
    - N (int): Number of time steps in the simulation.
    - r (float): Risk-free interest rate.

    Methods:
    - generate_paths(self): Abstract method that must be implemented by subclasses to generate stock price paths.
    - plot(self, paths, title="Stock Price Paths"): Plots the stock price paths.
    """
    def __init__(self, S0, T, N, r):
        self.S0 = S0  # Initial stock price
        self.T = T    # Time horizon
        self.N = N    # Number of time steps
        self.r = r    # Risk-free rate
        self.dt = T / N  # Time increment

    @staticmethod
    def _make_rng(random_seed=None):
        if random_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(random_seed))

    def generate_paths(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def plot(self, paths, title="Stock Price Paths"):
        plt.figure(figsize=(10, 6))
        plt.plot(paths.numpy().T)
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.show()

class GBMStock(Stock):
    """
    A subclass of Stock that models stock prices using the Geometric Brownian Motion (GBM) model.

    Arguments:
    - S0 (float): Initial stock price.
    - T (float): Time horizon for the simulation.
    - N (int): Number of time steps in the simulation.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the stock.

    Methods:
    - generate_paths(self, num_paths): Generates stock price paths using the GBM model.
    """
    def __init__(
        self,
        S0,
        T,
        N,
        r,
        sigma,
        sigma_per_path_mode="fixed",
        sigma_uniform_low=None,
        sigma_uniform_high=None,
        sigma_discrete_values=None,
        sigma_discrete_probs=None,
    ):
        super().__init__(S0, T, N, r)
        self.sigma = sigma  # Stock volatility
        mode = str(sigma_per_path_mode).strip().lower()
        if mode not in {"fixed", "uniform", "discrete"}:
            raise ValueError("sigma_per_path_mode must be one of {'fixed','uniform','discrete'}.")
        self.sigma_per_path_mode = mode
        self.sigma_uniform_low = sigma_uniform_low
        self.sigma_uniform_high = sigma_uniform_high
        self.sigma_discrete_values = sigma_discrete_values
        self.sigma_discrete_probs = sigma_discrete_probs
        self._last_sampled_sigmas = None

    def _sample_sigma_vector(self, num_paths, rng):
        n = int(num_paths)
        base_sigma = float(self.sigma)
        mode = self.sigma_per_path_mode
        if mode == "fixed":
            if base_sigma <= 0.0:
                raise ValueError("sigma must be > 0 for sigma_per_path_mode='fixed'.")
            return np.full((n,), base_sigma, dtype=np.float64)

        if mode == "uniform":
            lo = self.sigma_uniform_low
            hi = self.sigma_uniform_high
            if lo is None or hi is None:
                raise ValueError(
                    "sigma_uniform_low and sigma_uniform_high are required for sigma_per_path_mode='uniform'."
                )
            lo = float(lo)
            hi = float(hi)
            if lo <= 0.0 or hi <= 0.0 or lo >= hi:
                raise ValueError("Require 0 < sigma_uniform_low < sigma_uniform_high.")
            return rng.uniform(lo, hi, size=n).astype(np.float64)

        values = self.sigma_discrete_values
        probs = self.sigma_discrete_probs
        if values is None or len(values) == 0:
            raise ValueError(
                "sigma_discrete_values (non-empty) is required for sigma_per_path_mode='discrete'."
            )
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(vals)) or np.any(vals <= 0.0):
            raise ValueError("sigma_discrete_values must be finite and > 0.")
        p = None
        if probs is not None:
            p = np.asarray(probs, dtype=np.float64).reshape(-1)
            if p.shape[0] != vals.shape[0]:
                raise ValueError(
                    "sigma_discrete_probs length mismatch: expected "
                    f"{vals.shape[0]}, got {p.shape[0]}."
                )
            if np.any(~np.isfinite(p)) or np.any(p < 0.0):
                raise ValueError("sigma_discrete_probs must be finite and >= 0.")
            total = float(np.sum(p))
            if total <= 0.0:
                raise ValueError("sigma_discrete_probs must sum to a positive value.")
            p = p / total
        idx = rng.choice(vals.shape[0], size=n, replace=True, p=p)
        return vals[idx]

    def get_last_sampled_sigmas(self):
        if self._last_sampled_sigmas is None:
            return None
        return self._last_sampled_sigmas.copy()

    def _simulate_paths(self, num_paths, n_steps, random_seed=None):
        dt = self.dt
        S0 = self.S0
        r = self.r

        rng = self._make_rng(random_seed)
        sigma_vec = self._sample_sigma_vector(num_paths=num_paths, rng=rng)
        self._last_sampled_sigmas = sigma_vec.astype(np.float32)

        # Generate random normal variables for the Brownian motion increments.
        dW = rng.normal(0.0, 1.0, size=(num_paths, n_steps)) * np.sqrt(dt)
        drift = (r - 0.5 * np.square(sigma_vec[:, None])) * dt
        increments = drift + sigma_vec[:, None] * dW

        # Vectorized GBM: log S_t = log S0 + cumulative increments.
        log_cum = np.cumsum(increments, axis=1)
        log_full = np.concatenate([np.zeros((num_paths, 1)), log_cum], axis=1)
        return S0 * np.exp(log_full)

    def generate_paths(self, num_paths, random_seed = None):
        """
        Generates stock price paths using the GBM model.

        Arguments:
        - num_paths (int): Number of paths to simulate.
        - random_seed (int, optional): Seed for random number generation. Default is None.

        Returns:
        - S_paths (tf.Tensor): A TensorFlow tensor containing the generated stock price paths.
          Shape is (num_paths, N+1).
        """
        S = self._simulate_paths(num_paths=num_paths, n_steps=self.N, random_seed=random_seed)
        return tf.convert_to_tensor(S, dtype=tf.float32)

    def generate_paths_with_context(self, num_paths, n_context_steps, random_seed=None):
        """
        Generate GBM paths with explicit pre-history.

        Returns:
        - tf.Tensor of shape (num_paths, n_context_steps + N + 1), where:
          * first n_context_steps points are pre-history,
          * last N+1 points are the hedge window (S0..SN).
        """
        n_context_steps = int(n_context_steps)
        if n_context_steps < 0:
            raise ValueError("n_context_steps must be >= 0.")
        n_total_steps = int(self.N + n_context_steps)
        S = self._simulate_paths(num_paths=num_paths, n_steps=n_total_steps, random_seed=random_seed)
        return tf.convert_to_tensor(S, dtype=tf.float32)

class HestonStock(Stock):
    """
    A subclass of Stock that models stock prices using the Heston stochastic volatility model.

    Arguments:
    - S0 (float): Initial stock price.
    - T (float): Time horizon for the simulation.
    - N (int): Number of time steps in the simulation.
    - r (float): Risk-free interest rate.
    - v0 (float): Initial variance.
    - kappa (float): Rate of mean reversion for the variance.
    - theta (float): Long-term variance (mean level).
    - xi (float): Volatility of variance (volatility of volatility).
    - rho (float): Correlation between the Brownian motions driving the stock price and variance.
    - return_variance(bool): True
    Methods:
    - generate_paths(self, num_paths): Generates stock price paths using the Heston model,
      with an option to return the variance paths.
    """
    def __init__(self, S0, T, N, r, v0, kappa, theta, xi, rho, return_variance=True):
        super().__init__(S0, T, N, r)
        self.v0 = v0        # Initial variance
        self.kappa = kappa  # Rate of reversion
        self.theta = theta  # Long-term variance
        self.xi = xi        # Volatility of variance
        self.rho = rho      # Correlation between Brownian motions
        self.return_variance = return_variance

    def generate_paths(self, num_paths, random_seed = None):
        """
        Generates stock price paths using the Heston model, with an option to return the variance paths.

        Arguments:
        - num_paths (int): Number of paths to simulate.
        - return_variance (bool, optional): If True, the method returns both the stock price paths and the variance paths.
          Default is False.
        - random_seed (int, optional): Seed for random number generation. Default is None.

        Returns:
        - S_paths (tf.Tensor): A TensorFlow tensor containing the generated stock price paths. Shape is (num_paths, N+1).
        - v_paths (tf.Tensor, optional): A TensorFlow tensor containing the generated variance paths,
          returned if return_variance is True. Shape is (num_paths, N+1).
        """
        dt = self.dt
        S0 = self.S0
        r = self.r
        v0 = self.v0
        kappa = self.kappa
        theta = self.theta
        xi = self.xi
        rho = self.rho

        rng = self._make_rng(random_seed)

        # Generate correlated random normal variables
        dW1 = rng.normal(0.0, 1.0, size=(num_paths, self.N)) * np.sqrt(dt)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * rng.normal(0.0, 1.0, size=(num_paths, self.N)) * np.sqrt(dt)

        # Initialize the paths for stock prices and variances
        S = np.zeros((num_paths, self.N + 1))
        v = np.zeros((num_paths, self.N + 1))
        S[:, 0] = S0
        v[:, 0] = v0

        # Generate the paths using the Heston model
        for t in range(1, self.N + 1):
            v[:, t] = v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + xi * np.sqrt(v[:, t-1]) * dW2[:, t-1]
            v[:, t] = np.maximum(v[:, t], 0)  # Ensure variance stays non-negative

            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t]) * dW1[:, t-1])

        # Convert the paths to TensorFlow tensors
        S_paths = tf.convert_to_tensor(S, dtype=tf.float32)
        v_paths = tf.convert_to_tensor(v, dtype=tf.float32)

        if self.return_variance:
            return S_paths, v_paths
        else:
            return S_paths
