"""Training utilities for DDPM denoisers."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from DeepHedging.utils.diffusion_schedule import DiffusionSchedule


def _gather_by_t(values: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """Gather schedule values per-sample and reshape for broadcasting."""

    gathered = tf.gather(values, t)
    return tf.reshape(gathered, [-1, 1, 1])


def _build_optimizer(learning_rate: float, weight_decay: float) -> tf.keras.optimizers.Optimizer:
    if weight_decay > 0:
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _validate_training_inputs(
    x_train: np.ndarray,
    schedule: DiffusionSchedule,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
):
    arr = np.asarray(x_train, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"x_train must be 3D (n_windows, seq_len, n_features). Got {arr.shape}.")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ValueError("x_train dimensions must be > 0.")
    if not np.isfinite(arr).all():
        raise ValueError("x_train contains NaN or inf.")
    if schedule.timesteps <= 1:
        raise ValueError("schedule.timesteps must be > 1.")
    if epochs <= 0:
        raise ValueError("train_epochs must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0.")
    if grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be > 0.")


def train_diffusion_denoiser(
    model: tf.keras.Model,
    x_train: np.ndarray,
    schedule: DiffusionSchedule,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    use_ema: bool,
    ema_decay: float,
    random_seed: int | None = None,
    verbose: int = 1,
) -> tuple[list[dict[str, Any]], list[np.ndarray] | None]:
    """Train epsilon-predictor with standard DDPM objective."""

    _validate_training_inputs(
        x_train=x_train,
        schedule=schedule,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
    )
    if use_ema and not (0.0 < ema_decay < 1.0):
        raise ValueError("ema_decay must satisfy 0 < ema_decay < 1 when use_ema=True.")

    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    x_train = np.asarray(x_train, dtype=np.float32)
    n_samples = int(x_train.shape[0])
    optimizer = _build_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)

    sqrt_alphas_cumprod = tf.constant(schedule.sqrt_alphas_cumprod, dtype=tf.float32)
    sqrt_one_minus_alphas_cumprod = tf.constant(schedule.sqrt_one_minus_alphas_cumprod, dtype=tf.float32)
    n_steps = int(schedule.timesteps)

    dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(buffer_size=n_samples, seed=random_seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    ema_weights = None
    if use_ema:
        ema_weights = [np.array(w.numpy(), copy=True) for w in model.weights]

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        batch_losses: list[float] = []
        for x0 in dataset:
            bs = tf.shape(x0)[0]
            t = tf.random.uniform(shape=(bs,), minval=0, maxval=n_steps, dtype=tf.int32)
            noise = tf.random.normal(shape=tf.shape(x0), dtype=x0.dtype)

            sqrt_ab = _gather_by_t(sqrt_alphas_cumprod, t)
            sqrt_omb = _gather_by_t(sqrt_one_minus_alphas_cumprod, t)
            x_t = sqrt_ab * x0 + sqrt_omb * noise

            with tf.GradientTape() as tape:
                eps_pred = model([x_t, t], training=True)
                loss = tf.reduce_mean(tf.square(noise - eps_pred))

            grads = tape.gradient(loss, model.trainable_variables)
            clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=grad_clip_norm)
            optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

            if use_ema and ema_weights is not None:
                for idx, var in enumerate(model.weights):
                    ema_weights[idx] = ema_decay * ema_weights[idx] + (1.0 - ema_decay) * var.numpy()

            batch_losses.append(float(loss.numpy()))

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history.append({"epoch": int(epoch), "loss_mse_eps": epoch_loss})
        if verbose:
            print(f"[ddpm] epoch {epoch}/{epochs} loss={epoch_loss:.6f}")

    return history, ema_weights


def apply_ema_weights(model: tf.keras.Model, ema_weights: list[np.ndarray] | None) -> list[np.ndarray] | None:
    """Replace model weights by EMA and return original weights."""

    if ema_weights is None:
        return None
    if len(model.weights) != len(ema_weights):
        raise ValueError("EMA weights length mismatch.")
    original = [np.array(w.numpy(), copy=True) for w in model.weights]
    for var, value in zip(model.weights, ema_weights):
        var.assign(value)
    return original


def restore_weights(model: tf.keras.Model, weights: list[np.ndarray] | None) -> None:
    if weights is None:
        return
    if len(model.weights) != len(weights):
        raise ValueError("Weights length mismatch while restoring.")
    for var, value in zip(model.weights, weights):
        var.assign(value)
