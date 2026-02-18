"""Sampling utilities for DDPM/DDIM."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from DeepHedging.utils.diffusion_schedule import DiffusionSchedule


def _scalar_at(values: np.ndarray, t: int) -> tf.Tensor:
    return tf.constant(float(values[t]), dtype=tf.float32)


def _predict_x0(
    x_t: tf.Tensor,
    eps_pred: tf.Tensor,
    alpha_bar_t: tf.Tensor,
    sqrt_one_minus_alpha_bar_t: tf.Tensor,
) -> tf.Tensor:
    return (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / tf.sqrt(tf.maximum(alpha_bar_t, 1e-12))


def sample_ddpm(
    model: tf.keras.Model,
    schedule: DiffusionSchedule,
    n_samples: int,
    seq_len: int,
    n_features: int,
    random_seed: int | None = None,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if seq_len <= 0 or n_features <= 0:
        raise ValueError("seq_len and n_features must be > 0.")
    if random_seed is not None:
        tf.random.set_seed(int(random_seed))

    x = tf.random.normal(shape=(n_samples, seq_len, n_features), dtype=tf.float32)
    timesteps = int(schedule.timesteps)

    for t in range(timesteps - 1, -1, -1):
        t_batch = tf.fill([n_samples], t)
        eps_pred = model([x, t_batch], training=False)

        alpha_t = _scalar_at(schedule.alphas, t)
        alpha_bar_t = _scalar_at(schedule.alphas_cumprod, t)
        sqrt_recip_alpha_t = _scalar_at(schedule.sqrt_recip_alphas, t)
        sqrt_one_minus_alpha_bar_t = _scalar_at(schedule.sqrt_one_minus_alphas_cumprod, t)
        beta_t = _scalar_at(schedule.betas, t)

        x0_pred = _predict_x0(
            x_t=x,
            eps_pred=eps_pred,
            alpha_bar_t=alpha_bar_t,
            sqrt_one_minus_alpha_bar_t=sqrt_one_minus_alpha_bar_t,
        )
        mean = sqrt_recip_alpha_t * (x - (beta_t / tf.maximum(sqrt_one_minus_alpha_bar_t, 1e-12)) * eps_pred)

        if t > 0:
            variance = _scalar_at(schedule.posterior_variance, t)
            noise = tf.random.normal(shape=tf.shape(x), dtype=tf.float32)
            x = mean + tf.sqrt(tf.maximum(variance, 1e-12)) * noise
        else:
            x = x0_pred

    return x.numpy().astype(np.float32)


def sample_ddim(
    model: tf.keras.Model,
    schedule: DiffusionSchedule,
    n_samples: int,
    seq_len: int,
    n_features: int,
    sample_steps: int,
    eta: float = 0.0,
    random_seed: int | None = None,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if sample_steps <= 0:
        raise ValueError("sample_steps must be > 0.")
    if sample_steps > schedule.timesteps:
        raise ValueError("sample_steps must be <= schedule.timesteps.")
    if not (0.0 <= float(eta) <= 1.0):
        raise ValueError("ddim_eta must satisfy 0 <= eta <= 1.")
    if random_seed is not None:
        tf.random.set_seed(int(random_seed))

    x = tf.random.normal(shape=(n_samples, seq_len, n_features), dtype=tf.float32)
    time_grid = np.linspace(schedule.timesteps - 1, 0, num=sample_steps, dtype=np.int64)

    for idx, t in enumerate(time_grid):
        t_int = int(t)
        t_batch = tf.fill([n_samples], t_int)
        eps_pred = model([x, t_batch], training=False)

        alpha_bar_t = _scalar_at(schedule.alphas_cumprod, t_int)
        sqrt_one_minus_alpha_bar_t = _scalar_at(schedule.sqrt_one_minus_alphas_cumprod, t_int)
        x0_pred = _predict_x0(
            x_t=x,
            eps_pred=eps_pred,
            alpha_bar_t=alpha_bar_t,
            sqrt_one_minus_alpha_bar_t=sqrt_one_minus_alpha_bar_t,
        )

        if idx == len(time_grid) - 1:
            x = x0_pred
            continue

        t_next = int(time_grid[idx + 1])
        alpha_bar_next = _scalar_at(schedule.alphas_cumprod, t_next)

        sigma = float(eta) * tf.sqrt(
            tf.maximum(
                (1.0 - alpha_bar_next) / tf.maximum(1.0 - alpha_bar_t, 1e-12)
                * (1.0 - alpha_bar_t / tf.maximum(alpha_bar_next, 1e-12)),
                0.0,
            )
        )
        direction = tf.sqrt(tf.maximum(1.0 - alpha_bar_next - tf.square(sigma), 0.0)) * eps_pred
        noise = tf.random.normal(shape=tf.shape(x), dtype=tf.float32) if float(eta) > 0 else 0.0
        x = tf.sqrt(tf.maximum(alpha_bar_next, 1e-12)) * x0_pred + direction + sigma * noise

    return x.numpy().astype(np.float32)
