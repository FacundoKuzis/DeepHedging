from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from DeepHedging.HedgingInstruments.stock import Stock
from DeepHedging.utils.diffusion_model import SinusoidalTimeEmbedding, build_diffusion_denoiser
from DeepHedging.utils.diffusion_sampling import sample_ddim, sample_ddpm
from DeepHedging.utils.diffusion_schedule import DiffusionSchedule, build_diffusion_schedule
from DeepHedging.utils.diffusion_training import apply_ema_weights, restore_weights, train_diffusion_denoiser
from DeepHedging.utils.market_data import download_ohlcv_to_csv, load_prices_csv
from DeepHedging.utils.timegan_features import (
    build_return_feature_matrix,
    feature_stats_table,
    make_sliding_windows,
    validate_feature_mode,
)
from DeepHedging.utils.timegan_transforms import (
    fit_transform_1d,
    inverse_transform_1d,
    validate_transform_name,
)


class DiffusionStock(Stock):
    """Stock instrument backed by DDPM/DDIM diffusion model on return windows."""

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
        model_dir: Optional[str] = None,
        download_if_missing: bool = True,
        retrain: bool = False,
        stride: int = 1,
        random_seed: int = 42,
        min_windows: int = 200,
        train_epochs: int = 300,
        batch_size: int = 128,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        training_target: str = "log_returns",
        return_transform: str = "gaussian_cdf",
        transform_eps: float = 1e-6,
        feature_mode: str = "returns_only",
        feature_rolling_vol_window: int = 5,
        legacy_return_clip: bool = False,
        diffusion_steps: int = 200,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        model_hidden_dim: int = 128,
        model_num_res_blocks: int = 4,
        model_dropout: float = 0.1,
        time_embedding_dim: int = 64,
        sampler_type: str = "ddpm",
        sample_steps: int = 200,
        ddim_eta: float = 0.0,
    ):
        super().__init__(S0=S0, T=T, N=N, r=r)

        if N <= 0:
            raise ValueError("N must be > 0.")
        if stride <= 0:
            raise ValueError("stride must be > 0.")
        if min_windows <= 0:
            raise ValueError("min_windows must be > 0.")
        if not (0.0 < float(transform_eps) < 0.5):
            raise ValueError("transform_eps must satisfy 0 < transform_eps < 0.5.")
        if feature_rolling_vol_window <= 0:
            raise ValueError("feature_rolling_vol_window must be > 0.")
        if diffusion_steps <= 1:
            raise ValueError("diffusion_steps must be > 1.")
        if sample_steps <= 0:
            raise ValueError("sample_steps must be > 0.")
        if sample_steps > diffusion_steps:
            raise ValueError("sample_steps must be <= diffusion_steps.")
        if grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0.")

        training_target = str(training_target).strip().lower()
        if training_target != "log_returns":
            raise ValueError("DiffusionStock v1 supports only training_target='log_returns'.")

        sampler_type = str(sampler_type).strip().lower()
        if sampler_type not in {"ddpm", "ddim"}:
            raise ValueError("sampler_type must be one of {'ddpm','ddim'}.")
        beta_schedule = str(beta_schedule).strip().lower()
        if beta_schedule not in {"linear", "cosine"}:
            raise ValueError("beta_schedule must be one of {'linear','cosine'}.")

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.price_col = price_col
        self.download_if_missing = bool(download_if_missing)
        self.retrain = bool(retrain)
        self.stride = int(stride)
        self.random_seed = int(random_seed)
        self.min_windows = int(min_windows)
        self.training_target = training_target

        self.train_epochs = int(train_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)

        self.return_transform = validate_transform_name(return_transform)
        self.transform_eps = float(transform_eps)
        self.feature_mode = validate_feature_mode(feature_mode)
        self.feature_rolling_vol_window = int(feature_rolling_vol_window)
        self.legacy_return_clip = bool(legacy_return_clip)

        self.diffusion_steps = int(diffusion_steps)
        self.beta_schedule = beta_schedule
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.model_hidden_dim = int(model_hidden_dim)
        self.model_num_res_blocks = int(model_num_res_blocks)
        self.model_dropout = float(model_dropout)
        self.time_embedding_dim = int(time_embedding_dim)
        self.sampler_type = sampler_type
        self.sample_steps = int(sample_steps)
        self.ddim_eta = float(ddim_eta)

        self.csv_path = self._resolve_csv_path(csv_path)
        self.model_dir = self._resolve_model_dir(model_dir)
        self.model_path = os.path.join(self.model_dir, "diffusion_denoiser.keras")
        self.state_path = os.path.join(self.model_dir, "diffusion_state.npz")
        self.metadata_path = os.path.join(self.model_dir, "diffusion_metadata.json")
        self.ema_weights_path = os.path.join(self.model_dir, "diffusion_ema.weights.h5")

        self._df_prices = self._load_or_download_csv()
        self._series = self._extract_price_series(self._df_prices, self.price_col)
        self._real_windows = self._prepare_training_windows()
        self._log_returns = np.diff(np.log(np.maximum(self._series, 1e-8))).astype(np.float32)
        if self._log_returns.size < max(2, self.N):
            raise ValueError(
                "Not enough return observations to train diffusion model. "
                f"Need at least {self.N}, got {self._log_returns.size}."
            )

        self._returns_for_training = np.asarray(self._log_returns, dtype=np.float32)
        self._returns_mean = float(np.mean(self._returns_for_training))
        self._returns_std = float(np.std(self._returns_for_training))
        if not np.isfinite(self._returns_std) or self._returns_std < 1e-8:
            self._returns_std = 1e-8

        self._training_tensor: Optional[np.ndarray] = None
        self._raw_feature_windows: Optional[np.ndarray] = None
        self._feature_transform_states: list[dict[str, Any]] = []
        self._train_feature_names: list[str] = []
        self._training_manifest_df = pd.DataFrame()
        self._training_history_df = pd.DataFrame()

        self._schedule: Optional[DiffusionSchedule] = None
        self._model: Optional[tf.keras.Model] = None
        self._ema_weights: Optional[list[np.ndarray]] = None

        self._prepare_training_tensor()

        # Compatibility with analytical agents expecting stock_model.sigma
        self.sigma = self._estimate_sigma_from_series(self._series)

    def _resolve_csv_path(self, csv_path: Optional[str]) -> str:
        if csv_path:
            return os.path.abspath(csv_path)
        filename = f"{self.ticker}_{self.start_date}_{self.end_date}_{self.interval}.csv"
        return os.path.abspath(os.path.join("assets", "csvs", "diffusion", filename.replace(":", "-")))

    def _resolve_model_dir(self, model_dir: Optional[str]) -> str:
        if model_dir:
            return os.path.abspath(model_dir)
        dirname = f"{self.ticker}_{self.start_date}_{self.end_date}_{self.interval}_N{self.N}"
        return os.path.abspath(os.path.join("assets", "models", "diffusion", dirname.replace(":", "-")))

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
                f"Not enough clean windows to train diffusion model. "
                f"Found {len(windows)} windows, min_windows={self.min_windows}."
            )
        return np.asarray(windows, dtype=np.float32)

    def _prepare_training_tensor(self) -> None:
        raw_matrix, feature_names = build_return_feature_matrix(
            self._returns_for_training,
            feature_mode=self.feature_mode,
            feature_rolling_vol_window=self.feature_rolling_vol_window,
        )
        raw_windows = make_sliding_windows(
            raw_matrix,
            seq_len=self.N,
            stride=self.stride,
            min_windows=self.min_windows,
        )
        transformed = np.zeros_like(raw_windows, dtype=np.float32)
        states: list[dict[str, Any]] = []
        for feat_idx in range(raw_windows.shape[2]):
            method = self.return_transform if feat_idx == 0 else "minmax"
            flat = raw_windows[:, :, feat_idx].reshape(-1)
            transformed_flat, state = fit_transform_1d(flat, method=method, eps=self.transform_eps)
            transformed[:, :, feat_idx] = transformed_flat.reshape(raw_windows.shape[0], raw_windows.shape[1])
            states.append(state)

        stats = feature_stats_table(raw_windows, transformed, feature_names)
        stats["fit_input_mode"] = "explicit_windows"
        stats["training_target"] = self.training_target
        stats["seq_len"] = int(self.N)
        stats["n_features"] = int(transformed.shape[2])
        stats["n_windows"] = int(transformed.shape[0])
        stats["raw_observations"] = int(raw_matrix.shape[0])
        stats["effective_stride"] = int(self.stride)
        stats["return_transform"] = self.return_transform
        stats["feature_mode"] = self.feature_mode
        stats["transform_eps"] = float(self.transform_eps)
        stats["feature_transform"] = [
            str(self.return_transform if i == 0 else "minmax")
            for i in range(len(stats))
        ]

        self._training_tensor = transformed.astype(np.float32)
        self._raw_feature_windows = raw_windows.astype(np.float32)
        self._feature_transform_states = states
        self._train_feature_names = [str(name) for name in feature_names]
        self._training_manifest_df = stats

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist()}
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, dict):
            return {str(k): DiffusionStock._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DiffusionStock._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _from_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            if "__ndarray__" in value:
                return np.asarray(value["__ndarray__"], dtype=np.float64)
            return {k: DiffusionStock._from_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DiffusionStock._from_jsonable(v) for v in value]
        return value

    def _build_metadata(self) -> dict[str, Any]:
        if self._training_tensor is None:
            raise RuntimeError("Training tensor is not prepared.")
        return {
            "version": 1,
            "ticker": self.ticker,
            "seq_len": int(self.N),
            "n_features": int(self._training_tensor.shape[2]),
            "feature_names": list(self._train_feature_names),
            "feature_mode": self.feature_mode,
            "feature_rolling_vol_window": int(self.feature_rolling_vol_window),
            "training_target": self.training_target,
            "return_transform": self.return_transform,
            "transform_eps": float(self.transform_eps),
            "stride": int(self.stride),
            "min_windows": int(self.min_windows),
            "sampler_type": self.sampler_type,
            "sample_steps": int(self.sample_steps),
            "ddim_eta": float(self.ddim_eta),
            "diffusion_steps": int(self.diffusion_steps),
            "beta_schedule": self.beta_schedule,
            "beta_start": float(self.beta_start),
            "beta_end": float(self.beta_end),
            "model_hidden_dim": int(self.model_hidden_dim),
            "model_num_res_blocks": int(self.model_num_res_blocks),
            "model_dropout": float(self.model_dropout),
            "time_embedding_dim": int(self.time_embedding_dim),
            "use_ema": bool(self.use_ema),
            "ema_decay": float(self.ema_decay),
            "feature_transform_states": self._to_jsonable(self._feature_transform_states),
        }

    @staticmethod
    def _schedule_from_npz(npz: Any) -> DiffusionSchedule:
        return DiffusionSchedule(
            timesteps=int(npz["timesteps"]),
            betas=np.asarray(npz["betas"], dtype=np.float32),
            alphas=np.asarray(npz["alphas"], dtype=np.float32),
            alphas_cumprod=np.asarray(npz["alphas_cumprod"], dtype=np.float32),
            alphas_cumprod_prev=np.asarray(npz["alphas_cumprod_prev"], dtype=np.float32),
            sqrt_alphas_cumprod=np.asarray(npz["sqrt_alphas_cumprod"], dtype=np.float32),
            sqrt_one_minus_alphas_cumprod=np.asarray(npz["sqrt_one_minus_alphas_cumprod"], dtype=np.float32),
            sqrt_recip_alphas=np.asarray(npz["sqrt_recip_alphas"], dtype=np.float32),
            posterior_variance=np.asarray(npz["posterior_variance"], dtype=np.float32),
            posterior_mean_coef1=np.asarray(npz["posterior_mean_coef1"], dtype=np.float32),
            posterior_mean_coef2=np.asarray(npz["posterior_mean_coef2"], dtype=np.float32),
        )

    def _save_model_bundle(self) -> None:
        if self._model is None or self._schedule is None:
            raise RuntimeError("Model/schedule are not available for saving.")
        os.makedirs(self.model_dir, exist_ok=True)

        self._model.save(self.model_path)
        np.savez(
            self.state_path,
            timesteps=np.array(self._schedule.timesteps, dtype=np.int64),
            betas=self._schedule.betas,
            alphas=self._schedule.alphas,
            alphas_cumprod=self._schedule.alphas_cumprod,
            alphas_cumprod_prev=self._schedule.alphas_cumprod_prev,
            sqrt_alphas_cumprod=self._schedule.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self._schedule.sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=self._schedule.sqrt_recip_alphas,
            posterior_variance=self._schedule.posterior_variance,
            posterior_mean_coef1=self._schedule.posterior_mean_coef1,
            posterior_mean_coef2=self._schedule.posterior_mean_coef2,
        )
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._build_metadata(), f, indent=2)

        if self.use_ema and self._ema_weights is not None:
            original = apply_ema_weights(self._model, self._ema_weights)
            try:
                self._model.save_weights(self.ema_weights_path)
            finally:
                restore_weights(self._model, original)

    def _load_model_bundle(self) -> None:
        if not (os.path.exists(self.model_path) and os.path.exists(self.state_path) and os.path.exists(self.metadata_path)):
            raise FileNotFoundError("Diffusion model artifacts are incomplete.")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if int(metadata.get("seq_len", -1)) != int(self.N):
            raise ValueError(
                f"Saved model seq_len={metadata.get('seq_len')} incompatible with N={self.N}."
            )
        if str(metadata.get("training_target", "")).strip().lower() != self.training_target:
            raise ValueError("Saved model training_target mismatch.")
        if str(metadata.get("feature_mode", "")).strip().lower() != self.feature_mode:
            raise ValueError(
                f"Saved model feature_mode={metadata.get('feature_mode')} "
                f"incompatible with requested feature_mode={self.feature_mode}."
            )
        if str(metadata.get("return_transform", "")).strip().lower() != self.return_transform:
            raise ValueError(
                f"Saved model return_transform={metadata.get('return_transform')} "
                f"incompatible with requested return_transform={self.return_transform}."
            )
        if int(metadata.get("n_features", -1)) != len(self._train_feature_names):
            raise ValueError(
                f"Saved model n_features={metadata.get('n_features')} incompatible with "
                f"requested n_features={len(self._train_feature_names)}."
            )

        self._model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"SinusoidalTimeEmbedding": SinusoidalTimeEmbedding},
            compile=False,
        )
        with np.load(self.state_path, allow_pickle=False) as npz:
            self._schedule = self._schedule_from_npz(npz)

        states = metadata.get("feature_transform_states", [])
        states = self._from_jsonable(states)
        self._feature_transform_states = [dict(x) for x in states]
        self._train_feature_names = [str(x) for x in metadata.get("feature_names", self._train_feature_names)]

        if os.path.exists(self.ema_weights_path):
            ema_model = build_diffusion_denoiser(
                seq_len=int(metadata["seq_len"]),
                n_features=int(metadata["n_features"]),
                hidden_dim=int(metadata["model_hidden_dim"]),
                num_res_blocks=int(metadata["model_num_res_blocks"]),
                dropout=float(metadata["model_dropout"]),
                time_embedding_dim=int(metadata["time_embedding_dim"]),
            )
            ema_model.load_weights(self.ema_weights_path)
            self._ema_weights = [np.array(w.numpy(), copy=True) for w in ema_model.weights]

    def _fit_or_load_model(self) -> None:
        if self._model is not None and self._schedule is not None:
            return

        if os.path.exists(self.model_path) and os.path.exists(self.state_path) and os.path.exists(self.metadata_path) and not self.retrain:
            self._load_model_bundle()
            return

        if self._training_tensor is None:
            self._prepare_training_tensor()
        if self._training_tensor is None or self._training_tensor.ndim != 3:
            raise RuntimeError("Training tensor is not prepared correctly.")
        if self._training_tensor.shape[0] < self.min_windows:
            raise ValueError(
                f"Training tensor has {self._training_tensor.shape[0]} windows, "
                f"below min_windows={self.min_windows}."
            )

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)

        self._schedule = build_diffusion_schedule(
            timesteps=self.diffusion_steps,
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )
        self._model = build_diffusion_denoiser(
            seq_len=int(self.N),
            n_features=int(self._training_tensor.shape[2]),
            hidden_dim=self.model_hidden_dim,
            num_res_blocks=self.model_num_res_blocks,
            dropout=self.model_dropout,
            time_embedding_dim=self.time_embedding_dim,
        )

        history_rows, ema_weights = train_diffusion_denoiser(
            model=self._model,
            x_train=self._training_tensor,
            schedule=self._schedule,
            epochs=self.train_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            random_seed=self.random_seed,
            verbose=1,
        )
        self._training_history_df = pd.DataFrame(history_rows)
        self._ema_weights = ema_weights
        self._save_model_bundle()

    def _sample_windows(self, num_paths: int, random_seed: Optional[int] = None) -> np.ndarray:
        self._fit_or_load_model()
        if self._model is None or self._schedule is None:
            raise RuntimeError("Diffusion model was not loaded or trained.")
        chosen_seed = self.random_seed if random_seed is None else int(random_seed)

        original_weights = None
        if self.use_ema and self._ema_weights is not None:
            original_weights = apply_ema_weights(self._model, self._ema_weights)
        try:
            if self.sampler_type == "ddpm":
                sampled = sample_ddpm(
                    model=self._model,
                    schedule=self._schedule,
                    n_samples=num_paths,
                    seq_len=self.N,
                    n_features=len(self._train_feature_names),
                    random_seed=chosen_seed,
                )
            else:
                sampled = sample_ddim(
                    model=self._model,
                    schedule=self._schedule,
                    n_samples=num_paths,
                    seq_len=self.N,
                    n_features=len(self._train_feature_names),
                    sample_steps=self.sample_steps,
                    eta=self.ddim_eta,
                    random_seed=chosen_seed,
                )
        finally:
            restore_weights(self._model, original_weights)
        return sampled

    def _stabilize_sampled_returns(self, sampled_returns: np.ndarray) -> np.ndarray:
        sampled_returns = np.asarray(sampled_returns, dtype=np.float32)
        if not np.isfinite(sampled_returns).all():
            raise ValueError("Sampled returns contain NaN or inf.")
        if self.legacy_return_clip:
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
        if sampled_windows.ndim != 3:
            raise ValueError(f"Expected 3D sampled windows, got shape {sampled_windows.shape}.")
        if sampled_windows.shape[1] != self.N:
            raise ValueError(
                f"Expected sampled windows with seq_len={self.N}, got {sampled_windows.shape}."
            )
        if sampled_windows.shape[2] < 1:
            raise ValueError("Sampled windows must have at least one feature channel.")
        if len(self._feature_transform_states) < 1:
            raise ValueError("Missing transform state for sampled returns decoding.")

        u_returns = sampled_windows[:, :, 0]
        decoded_returns = inverse_transform_1d(
            u_returns.reshape(-1),
            state=self._feature_transform_states[0],
            eps=self.transform_eps,
        ).reshape(u_returns.shape[0], u_returns.shape[1])
        decoded_returns = self._stabilize_sampled_returns(decoded_returns)
        return self._returns_to_price_paths(decoded_returns)

    def generate_paths(self, num_paths: int, random_seed: Optional[int] = None) -> tf.Tensor:
        if num_paths <= 0:
            raise ValueError("num_paths must be > 0.")
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
        sampled_windows = self._sample_windows(num_paths=num_paths, random_seed=random_seed)
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

    def get_training_manifest(self) -> pd.DataFrame:
        if self._training_manifest_df is None or self._training_manifest_df.empty:
            return pd.DataFrame()
        return self._training_manifest_df.copy()

    def get_training_history(self) -> pd.DataFrame:
        if self._training_history_df is None or self._training_history_df.empty:
            return pd.DataFrame()
        return self._training_history_df.copy()

    def get_noise_schedule(self) -> pd.DataFrame:
        self._fit_or_load_model()
        if self._schedule is None:
            return pd.DataFrame()
        return self._schedule.to_frame()

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
