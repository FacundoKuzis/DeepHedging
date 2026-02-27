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


class StudentTStock(Stock):
    """
    Stock simulator with Student-t innovations in log-return dynamics.

    Annualized log-return increment:
      lr_t = (r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * z_t
    where z_t follows a standardized Student-t(ν) with unit variance.
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
        student_t_df=8.0,
        student_t_df_per_path_mode="fixed",
        student_t_df_uniform_low=None,
        student_t_df_uniform_high=None,
        student_t_df_discrete_values=None,
        student_t_df_discrete_probs=None,
    ):
        super().__init__(S0, T, N, r)
        self.sigma = float(sigma)

        sigma_mode = str(sigma_per_path_mode).strip().lower()
        if sigma_mode not in {"fixed", "uniform", "discrete"}:
            raise ValueError("sigma_per_path_mode must be one of {'fixed','uniform','discrete'}.")
        self.sigma_per_path_mode = sigma_mode
        self.sigma_uniform_low = sigma_uniform_low
        self.sigma_uniform_high = sigma_uniform_high
        self.sigma_discrete_values = sigma_discrete_values
        self.sigma_discrete_probs = sigma_discrete_probs

        self.student_t_df = float(student_t_df)
        df_mode = str(student_t_df_per_path_mode).strip().lower()
        if df_mode not in {"fixed", "uniform", "discrete"}:
            raise ValueError(
                "student_t_df_per_path_mode must be one of {'fixed','uniform','discrete'}."
            )
        self.student_t_df_per_path_mode = df_mode
        self.student_t_df_uniform_low = student_t_df_uniform_low
        self.student_t_df_uniform_high = student_t_df_uniform_high
        self.student_t_df_discrete_values = student_t_df_discrete_values
        self.student_t_df_discrete_probs = student_t_df_discrete_probs

        self._last_sampled_sigmas = None
        self._last_sampled_student_t_df = None

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

    def _sample_student_t_df_vector(self, num_paths, rng):
        n = int(num_paths)
        mode = self.student_t_df_per_path_mode
        if mode == "fixed":
            df = float(self.student_t_df)
            if not np.isfinite(df) or df <= 2.0:
                raise ValueError("student_t_df must be finite and > 2 for fixed mode.")
            return np.full((n,), df, dtype=np.float64)

        if mode == "uniform":
            lo = self.student_t_df_uniform_low
            hi = self.student_t_df_uniform_high
            if lo is None or hi is None:
                raise ValueError(
                    "student_t_df_uniform_low and student_t_df_uniform_high are required "
                    "for student_t_df_per_path_mode='uniform'."
                )
            lo = float(lo)
            hi = float(hi)
            if lo <= 2.0 or hi <= 2.0 or lo >= hi:
                raise ValueError(
                    "Require 2 < student_t_df_uniform_low < student_t_df_uniform_high."
                )
            return rng.uniform(lo, hi, size=n).astype(np.float64)

        values = self.student_t_df_discrete_values
        probs = self.student_t_df_discrete_probs
        if values is None or len(values) == 0:
            raise ValueError(
                "student_t_df_discrete_values (non-empty) is required for "
                "student_t_df_per_path_mode='discrete'."
            )
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(vals)) or np.any(vals <= 2.0):
            raise ValueError("student_t_df_discrete_values must be finite and > 2.")
        p = None
        if probs is not None:
            p = np.asarray(probs, dtype=np.float64).reshape(-1)
            if p.shape[0] != vals.shape[0]:
                raise ValueError(
                    "student_t_df_discrete_probs length mismatch: expected "
                    f"{vals.shape[0]}, got {p.shape[0]}."
                )
            if np.any(~np.isfinite(p)) or np.any(p < 0.0):
                raise ValueError("student_t_df_discrete_probs must be finite and >= 0.")
            total = float(np.sum(p))
            if total <= 0.0:
                raise ValueError("student_t_df_discrete_probs must sum to a positive value.")
            p = p / total
        idx = rng.choice(vals.shape[0], size=n, replace=True, p=p)
        return vals[idx]

    def get_last_sampled_sigmas(self):
        if self._last_sampled_sigmas is None:
            return None
        return self._last_sampled_sigmas.copy()

    def get_last_sampled_student_t_df(self):
        if self._last_sampled_student_t_df is None:
            return None
        return self._last_sampled_student_t_df.copy()

    def _simulate_paths(self, num_paths, n_steps, random_seed=None):
        dt = float(self.dt)
        s0 = float(self.S0)
        r = float(self.r)
        n_paths = int(num_paths)
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0.")

        rng = self._make_rng(random_seed)
        sigma_vec = self._sample_sigma_vector(num_paths=n_paths, rng=rng)
        df_vec = self._sample_student_t_df_vector(num_paths=n_paths, rng=rng)
        self._last_sampled_sigmas = sigma_vec.astype(np.float32)
        self._last_sampled_student_t_df = df_vec.astype(np.float32)

        if n_steps == 0:
            return np.full((n_paths, 1), s0, dtype=np.float64)

        # Student-t by Gaussian / Chi-square ratio.
        normals = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
        chi2 = rng.chisquare(df=df_vec[:, None], size=(n_paths, n_steps))
        t_raw = normals / np.sqrt(np.maximum(chi2 / df_vec[:, None], 1e-12))
        # Standardize to unit variance (nu > 2).
        z = t_raw * np.sqrt((df_vec[:, None] - 2.0) / df_vec[:, None])

        drift = (r - 0.5 * np.square(sigma_vec[:, None])) * dt
        diffusion = sigma_vec[:, None] * np.sqrt(dt) * z
        increments = drift + diffusion
        log_cum = np.cumsum(increments, axis=1)
        log_full = np.concatenate([np.zeros((n_paths, 1), dtype=np.float64), log_cum], axis=1)
        return s0 * np.exp(log_full)

    def generate_paths(self, num_paths, random_seed=None):
        s = self._simulate_paths(num_paths=num_paths, n_steps=self.N, random_seed=random_seed)
        return tf.convert_to_tensor(s, dtype=tf.float32)

    def generate_paths_with_context(self, num_paths, n_context_steps, random_seed=None):
        n_context_steps = int(n_context_steps)
        if n_context_steps < 0:
            raise ValueError("n_context_steps must be >= 0.")
        n_total_steps = int(self.N + n_context_steps)
        s = self._simulate_paths(num_paths=num_paths, n_steps=n_total_steps, random_seed=random_seed)
        return tf.convert_to_tensor(s, dtype=tf.float32)


class GARCHStock(Stock):
    """
    GARCH(1,1)-style stock simulator in log-price space with optional leverage term.

    The conditional annualized variance follows:
      v_{t+1} = omega + alpha * eps_t^2 + beta * v_t + leverage * 1_{eps_t<0} * eps_t^2

    and log-returns are simulated as:
      lr_t = (r - 0.5 * v_t) * dt + sqrt(v_t * dt) * z_t
    """

    def __init__(
        self,
        S0,
        T,
        N,
        r,
        sigma,
        garch_alpha=0.05,
        garch_beta=0.9,
        garch_omega=None,
        garch_leverage=0.0,
        garch_use_student_t=False,
        garch_student_t_df=8.0,
        r_per_path_mode="fixed",
        r_uniform_low=None,
        r_uniform_high=None,
        r_discrete_values=None,
        r_discrete_probs=None,
        sigma_per_path_mode="fixed",
        sigma_uniform_low=None,
        sigma_uniform_high=None,
        sigma_discrete_values=None,
        sigma_discrete_probs=None,
        garch_alpha_per_path_mode="fixed",
        garch_alpha_uniform_low=None,
        garch_alpha_uniform_high=None,
        garch_alpha_discrete_values=None,
        garch_alpha_discrete_probs=None,
        garch_beta_per_path_mode="fixed",
        garch_beta_uniform_low=None,
        garch_beta_uniform_high=None,
        garch_beta_discrete_values=None,
        garch_beta_discrete_probs=None,
        garch_leverage_per_path_mode="fixed",
        garch_leverage_uniform_low=None,
        garch_leverage_uniform_high=None,
        garch_leverage_discrete_values=None,
        garch_leverage_discrete_probs=None,
        garch_student_t_df_per_path_mode="fixed",
        garch_student_t_df_uniform_low=None,
        garch_student_t_df_uniform_high=None,
        garch_student_t_df_discrete_values=None,
        garch_student_t_df_discrete_probs=None,
    ):
        super().__init__(S0, T, N, r)
        self.sigma = float(sigma)

        mode = str(sigma_per_path_mode).strip().lower()
        if mode not in {"fixed", "uniform", "discrete"}:
            raise ValueError("sigma_per_path_mode must be one of {'fixed','uniform','discrete'}.")
        self.sigma_per_path_mode = mode
        self.sigma_uniform_low = sigma_uniform_low
        self.sigma_uniform_high = sigma_uniform_high
        self.sigma_discrete_values = sigma_discrete_values
        self.sigma_discrete_probs = sigma_discrete_probs

        self.garch_alpha = float(garch_alpha)
        self.garch_beta = float(garch_beta)
        self.garch_omega = None if garch_omega is None else float(garch_omega)
        self.garch_leverage = float(garch_leverage)
        self.garch_use_student_t = bool(garch_use_student_t)
        self.garch_student_t_df = float(garch_student_t_df)

        self.r_per_path_mode = str(r_per_path_mode).strip().lower()
        self.r_uniform_low = r_uniform_low
        self.r_uniform_high = r_uniform_high
        self.r_discrete_values = r_discrete_values
        self.r_discrete_probs = r_discrete_probs

        self.garch_alpha_per_path_mode = str(garch_alpha_per_path_mode).strip().lower()
        self.garch_alpha_uniform_low = garch_alpha_uniform_low
        self.garch_alpha_uniform_high = garch_alpha_uniform_high
        self.garch_alpha_discrete_values = garch_alpha_discrete_values
        self.garch_alpha_discrete_probs = garch_alpha_discrete_probs

        self.garch_beta_per_path_mode = str(garch_beta_per_path_mode).strip().lower()
        self.garch_beta_uniform_low = garch_beta_uniform_low
        self.garch_beta_uniform_high = garch_beta_uniform_high
        self.garch_beta_discrete_values = garch_beta_discrete_values
        self.garch_beta_discrete_probs = garch_beta_discrete_probs

        self.garch_leverage_per_path_mode = str(garch_leverage_per_path_mode).strip().lower()
        self.garch_leverage_uniform_low = garch_leverage_uniform_low
        self.garch_leverage_uniform_high = garch_leverage_uniform_high
        self.garch_leverage_discrete_values = garch_leverage_discrete_values
        self.garch_leverage_discrete_probs = garch_leverage_discrete_probs

        self.garch_student_t_df_per_path_mode = str(garch_student_t_df_per_path_mode).strip().lower()
        self.garch_student_t_df_uniform_low = garch_student_t_df_uniform_low
        self.garch_student_t_df_uniform_high = garch_student_t_df_uniform_high
        self.garch_student_t_df_discrete_values = garch_student_t_df_discrete_values
        self.garch_student_t_df_discrete_probs = garch_student_t_df_discrete_probs

        for mname, mval in [
            ("r_per_path_mode", self.r_per_path_mode),
            ("garch_alpha_per_path_mode", self.garch_alpha_per_path_mode),
            ("garch_beta_per_path_mode", self.garch_beta_per_path_mode),
            ("garch_leverage_per_path_mode", self.garch_leverage_per_path_mode),
            ("garch_student_t_df_per_path_mode", self.garch_student_t_df_per_path_mode),
        ]:
            if mval not in {"fixed", "uniform", "discrete"}:
                raise ValueError(f"{mname} must be one of {'fixed','uniform','discrete'}.")

        if self.garch_alpha < 0.0:
            raise ValueError("garch_alpha must be >= 0.")
        if self.garch_beta < 0.0:
            raise ValueError("garch_beta must be >= 0.")
        if self.garch_alpha + self.garch_beta >= 1.0:
            raise ValueError("Require garch_alpha + garch_beta < 1 for stability.")
        stability_lhs = self.garch_alpha + self.garch_beta + 2.0 * self.garch_leverage
        if stability_lhs >= 1.0:
            raise ValueError(
                "Require garch_alpha + garch_beta + 2*garch_leverage < 1 for stationarity. "
                f"Got {stability_lhs:.6f}."
            )
        if self.garch_omega is not None and self.garch_omega <= 0.0:
            raise ValueError("garch_omega must be > 0 when provided.")
        if self.garch_use_student_t and self.garch_student_t_df <= 2.0:
            raise ValueError("garch_student_t_df must be > 2 when garch_use_student_t=true.")

        self._last_sampled_sigmas = None
        self._last_realized_sigmas = None
        self._last_sampled_rates = None
        self._last_sampled_garch_alpha = None
        self._last_sampled_garch_beta = None
        self._last_sampled_garch_leverage = None
        self._last_sampled_garch_student_t_df = None

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

    def get_last_sampled_rates(self):
        if self._last_sampled_rates is None:
            return None
        return self._last_sampled_rates.copy()

    def _sample_mode_vector(
        self,
        num_paths,
        rng,
        mode,
        fixed_value,
        uniform_low,
        uniform_high,
        discrete_values,
        discrete_probs,
        name,
        min_inclusive=None,
        min_exclusive=None,
    ):
        n = int(num_paths)
        mode = str(mode).strip().lower()

        if mode == "fixed":
            vals = np.full((n,), float(fixed_value), dtype=np.float64)
        elif mode == "uniform":
            if uniform_low is None or uniform_high is None:
                raise ValueError(
                    f"{name}_uniform_low and {name}_uniform_high are required for {name}_per_path_mode='uniform'."
                )
            lo = float(uniform_low)
            hi = float(uniform_high)
            if lo >= hi:
                raise ValueError(f"Require {name}_uniform_low < {name}_uniform_high.")
            vals = rng.uniform(lo, hi, size=n).astype(np.float64)
        elif mode == "discrete":
            if discrete_values is None or len(discrete_values) == 0:
                raise ValueError(
                    f"{name}_discrete_values (non-empty) is required for {name}_per_path_mode='discrete'."
                )
            candidates = np.asarray(discrete_values, dtype=np.float64).reshape(-1)
            if np.any(~np.isfinite(candidates)):
                raise ValueError(f"{name}_discrete_values must contain finite values.")
            probs = None
            if discrete_probs is not None:
                probs = np.asarray(discrete_probs, dtype=np.float64).reshape(-1)
                if probs.shape[0] != candidates.shape[0]:
                    raise ValueError(
                        f"{name}_discrete_probs length mismatch: expected {candidates.shape[0]}, got {probs.shape[0]}."
                    )
                if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
                    raise ValueError(f"{name}_discrete_probs must be finite and >= 0.")
                total = float(np.sum(probs))
                if total <= 0.0:
                    raise ValueError(f"{name}_discrete_probs must sum to a positive value.")
                probs = probs / total
            idx = rng.choice(candidates.shape[0], size=n, replace=True, p=probs)
            vals = candidates[idx]
        else:
            raise ValueError(f"{name}_per_path_mode must be one of {'fixed','uniform','discrete'}.")

        if min_inclusive is not None and np.any(vals < float(min_inclusive)):
            raise ValueError(f"{name} values must be >= {float(min_inclusive)}.")
        if min_exclusive is not None and np.any(vals <= float(min_exclusive)):
            raise ValueError(f"{name} values must be > {float(min_exclusive)}.")
        return vals

    def _draw_innovations(self, rng, num_paths, n_steps, df_vec=None):
        if not self.garch_use_student_t:
            return rng.normal(0.0, 1.0, size=(num_paths, n_steps))
        if df_vec is None:
            # Backward-compatible scalar df path.
            df = float(self.garch_student_t_df)
            z = rng.standard_t(df, size=(num_paths, n_steps))
            scale = np.sqrt((df - 2.0) / df)
            return z * scale
        # Pathwise Student-t innovations standardized to unit variance.
        df_vec = np.asarray(df_vec, dtype=np.float64).reshape(-1)
        normals = rng.normal(0.0, 1.0, size=(num_paths, n_steps))
        chi2 = rng.chisquare(df=df_vec[:, None], size=(num_paths, n_steps))
        t_raw = normals / np.sqrt(np.maximum(chi2 / df_vec[:, None], 1e-12))
        return t_raw * np.sqrt((df_vec[:, None] - 2.0) / df_vec[:, None])

    def _simulate_paths(self, num_paths, n_steps, random_seed=None):
        dt = float(self.dt)
        s0 = float(self.S0)
        n_paths = int(num_paths)
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0.")

        rng = self._make_rng(random_seed)
        r_vec = self._sample_mode_vector(
            num_paths=n_paths,
            rng=rng,
            mode=self.r_per_path_mode,
            fixed_value=float(self.r),
            uniform_low=self.r_uniform_low,
            uniform_high=self.r_uniform_high,
            discrete_values=self.r_discrete_values,
            discrete_probs=self.r_discrete_probs,
            name="r",
        )
        sigma_long_run = self._sample_sigma_vector(num_paths=n_paths, rng=rng)
        var_long_run = np.maximum(np.square(sigma_long_run), 1e-12)

        alpha = self._sample_mode_vector(
            num_paths=n_paths,
            rng=rng,
            mode=self.garch_alpha_per_path_mode,
            fixed_value=float(self.garch_alpha),
            uniform_low=self.garch_alpha_uniform_low,
            uniform_high=self.garch_alpha_uniform_high,
            discrete_values=self.garch_alpha_discrete_values,
            discrete_probs=self.garch_alpha_discrete_probs,
            name="garch_alpha",
            min_inclusive=0.0,
        )
        beta = self._sample_mode_vector(
            num_paths=n_paths,
            rng=rng,
            mode=self.garch_beta_per_path_mode,
            fixed_value=float(self.garch_beta),
            uniform_low=self.garch_beta_uniform_low,
            uniform_high=self.garch_beta_uniform_high,
            discrete_values=self.garch_beta_discrete_values,
            discrete_probs=self.garch_beta_discrete_probs,
            name="garch_beta",
            min_inclusive=0.0,
        )
        leverage = self._sample_mode_vector(
            num_paths=n_paths,
            rng=rng,
            mode=self.garch_leverage_per_path_mode,
            fixed_value=float(self.garch_leverage),
            uniform_low=self.garch_leverage_uniform_low,
            uniform_high=self.garch_leverage_uniform_high,
            discrete_values=self.garch_leverage_discrete_values,
            discrete_probs=self.garch_leverage_discrete_probs,
            name="garch_leverage",
            min_inclusive=0.0,
        )

        if np.any(alpha + beta >= 1.0):
            raise ValueError(
                "Sampled GARCH parameters violate alpha + beta < 1 for at least one path."
            )
        stability = alpha + beta + 2.0 * leverage
        if np.any(stability >= 1.0):
            raise ValueError(
                "Sampled GARCH parameters violate alpha + beta + 2*leverage < 1 for at least one path."
            )

        df_vec = None
        if self.garch_use_student_t:
            df_vec = self._sample_mode_vector(
                num_paths=n_paths,
                rng=rng,
                mode=self.garch_student_t_df_per_path_mode,
                fixed_value=float(self.garch_student_t_df),
                uniform_low=self.garch_student_t_df_uniform_low,
                uniform_high=self.garch_student_t_df_uniform_high,
                discrete_values=self.garch_student_t_df_discrete_values,
                discrete_probs=self.garch_student_t_df_discrete_probs,
                name="garch_student_t_df",
                min_exclusive=2.0,
            )

        if self.garch_omega is None:
            omega = np.maximum((1.0 - alpha - beta) * var_long_run, 1e-14)
        else:
            omega = np.full((n_paths,), float(self.garch_omega), dtype=np.float64)

        z = self._draw_innovations(
            rng=rng,
            num_paths=n_paths,
            n_steps=n_steps,
            df_vec=df_vec,
        )
        log_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)

        var_t = var_long_run.copy()
        var_accum = np.zeros((n_paths,), dtype=np.float64)
        for t in range(n_steps):
            var_t = np.maximum(var_t, 1e-12)
            var_accum += var_t

            shock = np.sqrt(var_t * dt) * z[:, t]
            drift = (r_vec - 0.5 * var_t) * dt
            log_paths[:, t + 1] = log_paths[:, t] + drift + shock

            eps = np.sqrt(var_t) * z[:, t]
            neg = (eps < 0.0).astype(np.float64)
            var_t = omega + alpha * np.square(eps) + leverage * neg * np.square(eps) + beta * var_t

        if n_steps > 0:
            sigma_effective = np.sqrt(np.maximum(var_accum / float(n_steps), 1e-12))
        else:
            sigma_effective = sigma_long_run.copy()
        # Store both notions:
        # - _last_sampled_sigmas: ex-ante per-path long-run sigma (used by env pricing pathwise)
        # - _last_realized_sigmas: realized effective sigma from simulated path variance
        self._last_sampled_sigmas = sigma_long_run.astype(np.float32)
        self._last_realized_sigmas = sigma_effective.astype(np.float32)
        self._last_sampled_rates = r_vec.astype(np.float32)
        self._last_sampled_garch_alpha = alpha.astype(np.float32)
        self._last_sampled_garch_beta = beta.astype(np.float32)
        self._last_sampled_garch_leverage = leverage.astype(np.float32)
        self._last_sampled_garch_student_t_df = (
            None if df_vec is None else df_vec.astype(np.float32)
        )

        paths = s0 * np.exp(log_paths)
        return paths

    def generate_paths(self, num_paths, random_seed=None):
        s = self._simulate_paths(num_paths=num_paths, n_steps=self.N, random_seed=random_seed)
        return tf.convert_to_tensor(s, dtype=tf.float32)

    def generate_paths_with_context(self, num_paths, n_context_steps, random_seed=None):
        n_context_steps = int(n_context_steps)
        if n_context_steps < 0:
            raise ValueError("n_context_steps must be >= 0.")
        n_total_steps = int(self.N + n_context_steps)
        s = self._simulate_paths(num_paths=num_paths, n_steps=n_total_steps, random_seed=random_seed)
        return tf.convert_to_tensor(s, dtype=tf.float32)


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
