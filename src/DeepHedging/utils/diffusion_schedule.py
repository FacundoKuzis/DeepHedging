"""Noise schedules for DDPM/DDIM time-series models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiffusionSchedule:
    """Precomputed diffusion coefficients."""

    timesteps: int
    betas: np.ndarray
    alphas: np.ndarray
    alphas_cumprod: np.ndarray
    alphas_cumprod_prev: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    sqrt_recip_alphas: np.ndarray
    posterior_variance: np.ndarray
    posterior_mean_coef1: np.ndarray
    posterior_mean_coef2: np.ndarray

    def to_frame(self):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise ImportError("pandas is required to export schedule tables.") from exc
        idx = np.arange(self.timesteps, dtype=np.int64)
        return pd.DataFrame(
            {
                "timestep": idx,
                "beta": self.betas,
                "alpha": self.alphas,
                "alpha_cumprod": self.alphas_cumprod,
                "alpha_cumprod_prev": self.alphas_cumprod_prev,
                "sqrt_alpha_cumprod": self.sqrt_alphas_cumprod,
                "sqrt_one_minus_alpha_cumprod": self.sqrt_one_minus_alphas_cumprod,
                "sqrt_recip_alpha": self.sqrt_recip_alphas,
                "posterior_variance": self.posterior_variance,
                "posterior_mean_coef1": self.posterior_mean_coef1,
                "posterior_mean_coef2": self.posterior_mean_coef2,
            }
        )


def _linear_betas(timesteps: int, beta_start: float, beta_end: float) -> np.ndarray:
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    return np.clip(betas, 1e-8, 0.999).astype(np.float64)


def _cosine_betas(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(timesteps + 1, dtype=np.float64)
    x = steps / float(timesteps)
    alphas_cumprod = np.cos(((x + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-8, 0.999).astype(np.float64)


def build_diffusion_schedule(
    timesteps: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
) -> DiffusionSchedule:
    """Create precomputed schedule constants."""

    if timesteps <= 1:
        raise ValueError("diffusion_steps must be > 1.")
    schedule_name = str(beta_schedule).strip().lower()
    if schedule_name not in {"linear", "cosine"}:
        raise ValueError("beta_schedule must be one of {'linear', 'cosine'}.")
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError("beta_start and beta_end must be > 0.")
    if beta_end <= beta_start and schedule_name == "linear":
        raise ValueError("For linear schedule, beta_end must be > beta_start.")

    if schedule_name == "linear":
        betas = _linear_betas(timesteps, beta_start=beta_start, beta_end=beta_end)
    else:
        betas = _cosine_betas(timesteps)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.concatenate(([1.0], alphas_cumprod[:-1]))
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(np.maximum(1.0 - alphas_cumprod, 1e-12))
    sqrt_recip_alphas = np.sqrt(1.0 / np.maximum(alphas, 1e-12))

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / np.maximum(1.0 - alphas_cumprod, 1e-12)
    )
    posterior_variance = np.clip(posterior_variance, 1e-12, None)
    posterior_mean_coef1 = (
        betas * np.sqrt(alphas_cumprod_prev) / np.maximum(1.0 - alphas_cumprod, 1e-12)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / np.maximum(1.0 - alphas_cumprod, 1e-12)
    )

    return DiffusionSchedule(
        timesteps=int(timesteps),
        betas=betas.astype(np.float32),
        alphas=alphas.astype(np.float32),
        alphas_cumprod=alphas_cumprod.astype(np.float32),
        alphas_cumprod_prev=alphas_cumprod_prev.astype(np.float32),
        sqrt_alphas_cumprod=sqrt_alphas_cumprod.astype(np.float32),
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod.astype(np.float32),
        sqrt_recip_alphas=sqrt_recip_alphas.astype(np.float32),
        posterior_variance=posterior_variance.astype(np.float32),
        posterior_mean_coef1=posterior_mean_coef1.astype(np.float32),
        posterior_mean_coef2=posterior_mean_coef2.astype(np.float32),
    )
