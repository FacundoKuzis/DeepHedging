"""Keras denoiser network for DDPM/DDIM time-series generation."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="DeepHedging")
class SinusoidalTimeEmbedding(tf.keras.layers.Layer):
    """Deterministic sinusoidal embedding for integer timesteps."""

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        self.embedding_dim = int(embedding_dim)

    def call(self, timesteps: tf.Tensor) -> tf.Tensor:
        t = tf.cast(tf.reshape(timesteps, [-1]), tf.float32)
        half_dim = self.embedding_dim // 2
        half_dim = tf.maximum(half_dim, 1)
        freqs = tf.exp(
            -tf.math.log(10000.0)
            * tf.cast(tf.range(half_dim), tf.float32)
            / tf.cast(tf.maximum(half_dim - 1, 1), tf.float32)
        )
        args = tf.expand_dims(t, axis=1) * tf.expand_dims(freqs, axis=0)
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=1)
        if self.embedding_dim % 2 == 1:
            emb = tf.pad(emb, paddings=[[0, 0], [0, 1]])
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim})
        return config


def _residual_block(
    x: tf.Tensor,
    time_emb: tf.Tensor,
    hidden_dim: int,
    dropout: float,
    name_prefix: str,
) -> tf.Tensor:
    h = tf.keras.layers.LayerNormalization(name=f"{name_prefix}_ln1")(x)
    h = tf.keras.layers.Activation("swish", name=f"{name_prefix}_act1")(h)
    h = tf.keras.layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        padding="same",
        name=f"{name_prefix}_conv1",
    )(h)

    t_proj = tf.keras.layers.Dense(hidden_dim, name=f"{name_prefix}_tproj")(time_emb)
    t_proj = tf.keras.layers.Reshape((1, hidden_dim), name=f"{name_prefix}_treshape")(t_proj)
    h = tf.keras.layers.Add(name=f"{name_prefix}_add_t")([h, t_proj])

    h = tf.keras.layers.Activation("swish", name=f"{name_prefix}_act2")(h)
    h = tf.keras.layers.Dropout(dropout, name=f"{name_prefix}_dropout")(h)
    h = tf.keras.layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        padding="same",
        name=f"{name_prefix}_conv2",
    )(h)
    return tf.keras.layers.Add(name=f"{name_prefix}_residual")([x, h])


def build_diffusion_denoiser(
    seq_len: int,
    n_features: int,
    hidden_dim: int,
    num_res_blocks: int,
    dropout: float,
    time_embedding_dim: int,
) -> tf.keras.Model:
    """Build a 1D convolutional epsilon-predictor."""

    if seq_len <= 0 or n_features <= 0:
        raise ValueError("seq_len and n_features must be > 0.")
    if hidden_dim <= 0:
        raise ValueError("model_hidden_dim must be > 0.")
    if num_res_blocks <= 0:
        raise ValueError("model_num_res_blocks must be > 0.")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("model_dropout must satisfy 0 <= dropout < 1.")
    if time_embedding_dim <= 0:
        raise ValueError("time_embedding_dim must be > 0.")

    x_in = tf.keras.Input(shape=(seq_len, n_features), dtype=tf.float32, name="x_t")
    t_in = tf.keras.Input(shape=(), dtype=tf.int32, name="t")

    t_emb = SinusoidalTimeEmbedding(time_embedding_dim, name="time_sinusoidal")(t_in)
    t_emb = tf.keras.layers.Dense(hidden_dim, activation="swish", name="time_dense_1")(t_emb)
    t_emb = tf.keras.layers.Dense(hidden_dim, activation="swish", name="time_dense_2")(t_emb)

    x = tf.keras.layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        padding="same",
        name="input_projection",
    )(x_in)

    for block_idx in range(num_res_blocks):
        x = _residual_block(
            x,
            time_emb=t_emb,
            hidden_dim=hidden_dim,
            dropout=dropout,
            name_prefix=f"resblock_{block_idx}",
        )

    x = tf.keras.layers.LayerNormalization(name="output_ln")(x)
    x = tf.keras.layers.Activation("swish", name="output_act")(x)
    out = tf.keras.layers.Conv1D(
        filters=n_features,
        kernel_size=1,
        padding="same",
        name="eps_hat",
    )(x)

    return tf.keras.Model(inputs=[x_in, t_in], outputs=out, name="diffusion_denoiser")
