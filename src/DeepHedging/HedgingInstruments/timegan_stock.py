import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from DeepHedging.HedgingInstruments.stock import Stock
from DeepHedging.utils.market_data import download_ohlcv_to_csv, load_prices_csv


class TimeGANStock(Stock):
    """
    Stock instrument backed by a TimeGAN synthesizer from ydata-synthetic.

    The public interface matches the other instruments:
    - generate_paths(num_paths, random_seed=None) -> tf.Tensor with shape (num_paths, N+1)
    """

    def __init__(
        self,
        S0: float,
        T: float,
        N: int,
        r: float,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        price_col: str = "Close",
        csv_path: Optional[str] = None,
        model_path: Optional[str] = None,
        download_if_missing: bool = True,
        retrain: bool = False,
        stride: int = 1,
        random_seed: int = 42,
        min_windows: int = 200,
        train_epochs: int = 1_000,
        batch_size: int = 128,
        noise_dim: int = 32,
        layers_dim: int = 128,
        latent_dim: int = 24,
        learning_rate: float = 5e-4,
        gamma: float = 1.0,
        training_target: str = "log_returns",
        match_return_moments: bool = True,
        input_return_clip_quantiles: Optional[tuple[float, float]] = (0.01, 0.99),
        output_return_clip_quantiles: Optional[tuple[float, float]] = (0.01, 0.99),
    ):
        super().__init__(S0=S0, T=T, N=N, r=r)

        if N <= 0:
            raise ValueError("N must be > 0.")
        if stride <= 0:
            raise ValueError("stride must be > 0.")
        if min_windows <= 0:
            raise ValueError("min_windows must be > 0.")

        training_target = str(training_target).strip().lower()
        if training_target not in {"log_returns", "price_levels"}:
            raise ValueError(
                "training_target must be one of {'log_returns', 'price_levels'}."
            )

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.price_col = price_col
        self.download_if_missing = download_if_missing
        self.retrain = retrain
        self.stride = stride
        self.random_seed = random_seed
        self.min_windows = min_windows

        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.layers_dim = layers_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.training_target = training_target
        self.match_return_moments = bool(match_return_moments)
        self.input_return_clip_quantiles = input_return_clip_quantiles
        self.output_return_clip_quantiles = output_return_clip_quantiles

        self.csv_path = self._resolve_csv_path(csv_path)
        self.model_path = self._resolve_model_path(model_path)

        self._df_prices = self._load_or_download_csv()
        self._series = self._extract_price_series(self._df_prices, self.price_col)
        self._real_windows = self._prepare_training_windows()
        self._log_returns = np.diff(np.log(np.maximum(self._series, 1e-8))).astype(np.float32)
        if self._log_returns.size < max(2, self.N):
            raise ValueError(
                "Not enough return observations to train TimeGAN on log_returns. "
                f"Need at least {self.N} returns, got {self._log_returns.size}."
            )

        self._returns_for_training = self._clip_by_quantiles(
            self._log_returns, self.input_return_clip_quantiles
        )
        self._returns_mean = float(np.mean(self._returns_for_training))
        self._returns_std = float(np.std(self._returns_for_training))
        if not np.isfinite(self._returns_std) or self._returns_std < 1e-8:
            self._returns_std = 1e-8

        self._return_clip_low = None
        self._return_clip_high = None
        if self.output_return_clip_quantiles is not None:
            low_q, high_q = self._validate_quantiles(self.output_return_clip_quantiles)
            self._return_clip_low = float(np.quantile(self._returns_for_training, low_q))
            self._return_clip_high = float(np.quantile(self._returns_for_training, high_q))

        self._returns_train_min = float(np.min(self._returns_for_training))
        self._returns_train_max = float(np.max(self._returns_for_training))
        self._price_train_min = float(np.min(self._series))
        self._price_train_max = float(np.max(self._series))

        self._model_sequence_length = self.N if self.training_target == "log_returns" else self.N + 1
        self._loaded_representation = self.training_target
        self._validate_training_capacity()
        self._synthesizer = None

        # Exposed to keep compatibility with analytical agents that read stock_model.sigma
        self.sigma = self._estimate_sigma_from_series(self._series)

    @staticmethod
    def _validate_quantiles(quantiles: tuple[float, float]) -> tuple[float, float]:
        if quantiles is None:
            raise ValueError("quantiles cannot be None.")
        if len(quantiles) != 2:
            raise ValueError("quantiles must have exactly 2 values (low, high).")
        low, high = float(quantiles[0]), float(quantiles[1])
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(
                f"Invalid quantiles {quantiles}. Expected 0 <= low < high <= 1."
            )
        return low, high

    def _clip_by_quantiles(
        self, values: np.ndarray, quantiles: Optional[tuple[float, float]]
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).copy()
        if quantiles is None:
            return arr
        low_q, high_q = self._validate_quantiles(quantiles)
        low = np.quantile(arr, low_q)
        high = np.quantile(arr, high_q)
        return np.clip(arr, low, high).astype(np.float32)

    def _validate_training_capacity(self) -> None:
        if self.training_target == "log_returns":
            series = self._returns_for_training
            label = "log-return"
        else:
            series = self._series
            label = "price"

        available = int(series.size - self._model_sequence_length + 1)
        if available < self.min_windows:
            raise ValueError(
                f"Not enough {label} windows to train TimeGAN with sequence_length={self._model_sequence_length}. "
                f"Available={available}, min_windows={self.min_windows}."
            )

    def _resolve_csv_path(self, csv_path: Optional[str]) -> str:
        if csv_path:
            return os.path.abspath(csv_path)
        filename = f"{self.ticker}_{self.start_date}_{self.end_date}_{self.interval}.csv"
        filename = filename.replace(":", "-")
        return os.path.abspath(os.path.join("assets", "csvs", "timegan", filename))

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        if model_path:
            return os.path.abspath(model_path)
        filename = f"{self.ticker}_{self.start_date}_{self.end_date}_{self.interval}_N{self.N}.pkl"
        filename = filename.replace(":", "-")
        return os.path.abspath(os.path.join("assets", "models", "timegan", filename))

    @staticmethod
    def _extract_price_series(df: pd.DataFrame, price_col: str) -> np.ndarray:
        col_map = {c.lower(): c for c in df.columns}
        selected_col = col_map.get(price_col.lower(), price_col)
        if selected_col not in df.columns:
            raise ValueError(
                f"Price column '{price_col}' not found. Available columns: {list(df.columns)}"
            )

        series = pd.to_numeric(df[selected_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        series = series[np.isfinite(series)]
        series = series[series > 0]
        if series.size == 0:
            raise ValueError(f"No positive prices found in '{selected_col}'.")
        return series

    def _load_or_download_csv(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            return load_prices_csv(self.csv_path, price_col=self.price_col)

        if not self.download_if_missing:
            raise FileNotFoundError(
                f"CSV not found and download_if_missing=False: {self.csv_path}"
            )

        download_ohlcv_to_csv(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
            output_csv=self.csv_path,
        )
        return load_prices_csv(self.csv_path, price_col=self.price_col)

    def _prepare_training_windows(self) -> np.ndarray:
        window_len = self.N + 1
        if self._series.size < window_len:
            raise ValueError(
                f"Not enough data to build windows of length {window_len}. "
                f"Got {self._series.size} rows."
            )

        windows = []
        for start in range(0, self._series.size - window_len + 1, self.stride):
            window = self._series[start : start + window_len]
            if np.isfinite(window).all() and (window > 0).all():
                windows.append(window)

        if len(windows) < self.min_windows:
            raise ValueError(
                f"Not enough clean windows to train TimeGAN. "
                f"Found {len(windows)} windows, min_windows={self.min_windows}."
            )

        return np.asarray(windows, dtype=np.float32)

    def _windows_to_long_format(self, windows: np.ndarray) -> pd.DataFrame:
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 2:
            raise ValueError(f"windows must be 2D, got shape {windows.shape}.")

        num_windows, window_len = windows.shape
        entity_id = np.repeat(np.arange(num_windows), window_len)
        time_idx = np.tile(np.arange(window_len), num_windows)
        close = windows.reshape(-1)

        return pd.DataFrame(
            {
                "entity_id": entity_id.astype(np.int64),
                "time_idx": time_idx.astype(np.int64),
                "close": close.astype(np.float32),
            }
        )

    def _require_ydata(self):
        try:
            from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
            from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
        except Exception as exc:  # pragma: no cover - exercised via integration only
            major, minor = sys.version_info.major, sys.version_info.minor
            raise ImportError(
                "TimeGANStock requires 'ydata-synthetic'. "
                "Install with `pip install ydata-synthetic` and use Python 3.11 "
                f"(current interpreter: {major}.{minor})."
            ) from exc
        return TimeSeriesSynthesizer, ModelParameters, TrainParameters

    def _fit_or_load_synthesizer(self):
        if self._synthesizer is not None:
            return self._synthesizer

        TimeSeriesSynthesizer, ModelParameters, TrainParameters = self._require_ydata()

        if os.path.exists(self.model_path) and not self.retrain:
            self._synthesizer = TimeSeriesSynthesizer.load(self.model_path)
            seq_len = int(getattr(self._synthesizer, "seq_len", self._model_sequence_length))
            if seq_len not in {self.N, self.N + 1}:
                raise ValueError(
                    f"Loaded TimeGAN model has unsupported sequence length={seq_len}. "
                    f"Expected {self.N} (returns) or {self.N + 1} (price levels)."
                )
            self._model_sequence_length = seq_len
            self._loaded_representation = "log_returns" if seq_len == self.N else "price_levels"
            return self._synthesizer

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)

        model_args = ModelParameters(
            batch_size=self.batch_size,
            lr=self.learning_rate,
            noise_dim=self.noise_dim,
            layers_dim=self.layers_dim,
            latent_dim=self.latent_dim,
            gamma=self.gamma,
        )
        train_args = TrainParameters(
            epochs=self.train_epochs,
            sequence_length=self._model_sequence_length,
            number_sequences=1,
        )

        if self.training_target == "log_returns":
            train_feature = self._returns_for_training
        else:
            train_feature = self._series
        train_df = pd.DataFrame({"feature": train_feature})

        synthesizer = TimeSeriesSynthesizer(
            modelname="timegan",
            model_parameters=model_args,
        )
        synthesizer.fit(
            train_df,
            train_arguments=train_args,
            num_cols=["feature"],
        )

        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        synthesizer.save(self.model_path)
        self._synthesizer = synthesizer
        self._loaded_representation = self.training_target
        return self._synthesizer

    def _sample_windows(self, num_paths: int) -> np.ndarray:
        synth = self._fit_or_load_synthesizer()
        seq_len = int(getattr(synth, "seq_len", self._model_sequence_length))
        if seq_len not in {self.N, self.N + 1}:
            raise ValueError(
                f"Sampled model sequence length={seq_len} is unsupported. "
                f"Expected {self.N} (returns) or {self.N + 1} (price levels)."
            )
        self._model_sequence_length = seq_len

        sampled = synth.sample(n_samples=num_paths)

        if not isinstance(sampled, list):
            raise ValueError(
                f"Unexpected sample type from TimeGAN: {type(sampled)}. Expected list."
            )
        if len(sampled) < num_paths:
            raise ValueError(
                f"TimeGAN returned {len(sampled)} samples, expected {num_paths}."
            )

        windows = np.zeros((num_paths, self._model_sequence_length), dtype=np.float32)
        for i in range(num_paths):
            sample_i = sampled[i]
            if isinstance(sample_i, pd.DataFrame):
                if sample_i.shape[1] < 1:
                    raise ValueError("Sampled DataFrame has no columns.")
                values = sample_i.iloc[:, 0].to_numpy(dtype=np.float32)
            else:
                values = np.asarray(sample_i, dtype=np.float32).reshape(-1)

            if values.shape[0] != self._model_sequence_length:
                raise ValueError(
                    f"Sample {i} has invalid length {values.shape[0]}, expected {self._model_sequence_length}."
                )
            windows[i] = values
        return windows

    @staticmethod
    def _inverse_minmax_scaling(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        values = np.clip(values, 0.0, 1.0)
        span = float(max_value - min_value)
        if not np.isfinite(span) or span < 1e-12:
            return np.full_like(values, fill_value=float(min_value), dtype=np.float32)
        return (values * span + float(min_value)).astype(np.float32)

    def _stabilize_sampled_returns(self, sampled_returns: np.ndarray) -> np.ndarray:
        sampled_returns = np.asarray(sampled_returns, dtype=np.float32)
        if not np.isfinite(sampled_returns).all():
            raise ValueError("Sampled returns contain NaN or inf.")

        if self.match_return_moments:
            synth_mean = float(np.mean(sampled_returns))
            synth_std = float(np.std(sampled_returns))
            if np.isfinite(synth_std) and synth_std > 1e-8:
                sampled_returns = (
                    (sampled_returns - synth_mean) * (self._returns_std / synth_std) + self._returns_mean
                ).astype(np.float32)
            else:
                sampled_returns = np.full_like(sampled_returns, self._returns_mean, dtype=np.float32)

        if self._return_clip_low is not None and self._return_clip_high is not None:
            sampled_returns = np.clip(sampled_returns, self._return_clip_low, self._return_clip_high)

        sampled_returns = np.clip(sampled_returns, -1.0, 1.0)
        return sampled_returns.astype(np.float32)

    def _returns_to_price_paths(self, sampled_returns: np.ndarray) -> np.ndarray:
        sampled_returns = np.asarray(sampled_returns, dtype=np.float32)
        if sampled_returns.ndim != 2 or sampled_returns.shape[1] != self.N:
            raise ValueError(
                f"Expected sampled returns with shape (num_paths, {self.N}), got {sampled_returns.shape}."
            )
        gross_returns = np.exp(sampled_returns)
        prices = np.empty((sampled_returns.shape[0], self.N + 1), dtype=np.float32)
        prices[:, 0] = self.S0
        prices[:, 1:] = self.S0 * np.cumprod(gross_returns, axis=1)
        prices = np.maximum(prices, 1e-8)
        prices[:, 0] = self.S0
        return prices.astype(np.float32)

    def _postprocess_to_prices(self, sampled_windows: np.ndarray) -> np.ndarray:
        sampled_windows = np.asarray(sampled_windows, dtype=np.float32)
        if sampled_windows.ndim != 2:
            raise ValueError(f"Expected 2D sampled windows, got shape {sampled_windows.shape}.")
        if not np.isfinite(sampled_windows).all():
            raise ValueError("Sampled windows contain NaN or inf.")

        if sampled_windows.shape[1] == self.N:
            # Preferred path: model learns log-returns, then we rebuild prices.
            sampled_returns = self._inverse_minmax_scaling(
                sampled_windows, self._returns_train_min, self._returns_train_max
            )
            sampled_returns = self._stabilize_sampled_returns(sampled_returns)
            return self._returns_to_price_paths(sampled_returns)

        if sampled_windows.shape[1] == self.N + 1:
            # Legacy path: model emits normalized price levels.
            denorm_levels = self._inverse_minmax_scaling(
                sampled_windows, self._price_train_min, self._price_train_max
            )
            denorm_levels = np.maximum(denorm_levels, 1e-8)
            first_values = np.maximum(denorm_levels[:, :1], 1e-8)
            scaled = self.S0 * (denorm_levels / first_values)
            scaled = np.maximum(scaled, 1e-8)
            scaled[:, 0] = self.S0
            return scaled.astype(np.float32)

        raise ValueError(
            f"Unexpected sampled window length={sampled_windows.shape[1]}. "
            f"Expected {self.N} (returns) or {self.N + 1} (price levels)."
        )

    def generate_paths(self, num_paths: int, random_seed: Optional[int] = None) -> tf.Tensor:
        if num_paths <= 0:
            raise ValueError("num_paths must be > 0.")

        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        sampled_windows = self._sample_windows(num_paths=num_paths)
        price_paths = self._postprocess_to_prices(sampled_windows)
        return tf.convert_to_tensor(price_paths, dtype=tf.float32)

    def get_real_windows(self, num_windows: int, random_seed: Optional[int] = None) -> np.ndarray:
        if num_windows <= 0:
            raise ValueError("num_windows must be > 0.")
        rng = np.random.default_rng(seed=random_seed)

        replace = num_windows > self._real_windows.shape[0]
        idx = rng.choice(self._real_windows.shape[0], size=num_windows, replace=replace)
        sampled = self._real_windows[idx].copy()
        first_values = np.maximum(sampled[:, :1], 1e-8)
        sampled = self.S0 * (sampled / first_values)
        sampled[:, 0] = self.S0
        return sampled.astype(np.float32)

    def _estimate_sigma_from_series(self, series: np.ndarray) -> float:
        if series.size < 2:
            return 0.2
        log_returns = np.diff(np.log(np.maximum(series, 1e-8)))
        if log_returns.size == 0:
            return 0.2
        sigma_step = float(np.nanstd(log_returns))
        if not np.isfinite(sigma_step) or sigma_step <= 0:
            return 0.2
        if self.dt <= 0:
            return sigma_step
        sigma = sigma_step / np.sqrt(self.dt)
        if not np.isfinite(sigma) or sigma <= 0:
            return 0.2
        return float(sigma)
