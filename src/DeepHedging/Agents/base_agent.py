import tensorflow as tf
from abc import ABC, abstractmethod
import os
import warnings
import inspect
import pickle


class BaseAgent(ABC):
    """
    The base class for all agents.

    Methods:
    - build_model(self): Abstract method to build the model architecture.
    - act(self, instrument_paths, T_minus_t): Abstract method to act based on inputs.
    - transform_input(self, *args): Optional method to transform inputs if necessary.
    """

    @abstractmethod
    def build_model(self):
        pass

    def transform_paths(self, instrument_paths, transformation_configs=None):
        """
        Transforms the instrument paths based on the specified transformation configurations.

        Arguments:
        - instrument_paths (tf.Tensor): The paths of the instrument. Shape: (batch_size, n_timesteps, n_instruments)
        - transformation_configs (list of dict, optional): List of dictionaries, each containing:
        - 'transformation_type' (str): The type of transformation ('log', 'log_moneyness').
        - 'K' (float, optional): The strike price, required if transformation_type is 'log_moneyness'.

        Returns:
        - transformed_paths (tf.Tensor): The transformed instrument paths. Shape: (batch_size, n_timesteps, n_instruments)
        """

        if transformation_configs is None:
            # No transformation, return the paths as-is
            return instrument_paths

        # Ensure the length of the list matches the number of instruments
        if len(transformation_configs) != instrument_paths.shape[-1]:
            raise ValueError("Length of transformation_configs list must match the number of instruments.")

        transformed_paths_list = []
        for i, config in enumerate(transformation_configs):
            if len(instrument_paths.shape) == 3:
                path_i = instrument_paths[:, :, i]  # Extract the path for the i-th instrument
            else:
                path_i = instrument_paths[:, i]  # Extract the path for the i-th instrument

            t_type = config.get('transformation_type')
            K = config.get('K')
            transformed_path_i = self._apply_transformation(path_i, t_type, K)
            transformed_paths_list.append(transformed_path_i)

        # Stack the transformed paths back into a single tensor
        transformed_paths = tf.stack(transformed_paths_list, axis=-1)
        return transformed_paths

    def _apply_transformation(self, instrument_paths, transformation_type, K=None):
        """
        Apply the specified transformation to the given instrument paths.

        Arguments:
        - instrument_paths (tf.Tensor): The paths of the instrument.
        - transformation_type (str or None): The type of transformation ('log', 'log_moneyness').
        - K (float, optional): The strike price, required if transformation_type is 'log_moneyness'.

        Returns:
        - transformed_paths (tf.Tensor): The transformed instrument paths.
        """
        paths = tf.convert_to_tensor(instrument_paths, dtype=tf.float32)
        # Numerical guard for heavy-tail simulators (e.g., stressed GARCH):
        # prevent log(0), log(inf) and overflow propagation into model inputs.
        eps = tf.constant(1e-8, dtype=tf.float32)
        max_price = tf.constant(1e12, dtype=tf.float32)
        safe_paths = tf.clip_by_value(paths, eps, max_price)

        if transformation_type is None:
            return paths
        elif transformation_type == 'log':
            return tf.math.log(safe_paths)
        elif transformation_type == 'log_moneyness':
            if K is None:
                raise ValueError("Strike price K must be provided for 'log moneyness' transformation.")
            strike = tf.constant(max(float(K), 1e-8), dtype=tf.float32)
            return tf.math.log(safe_paths / strike)
        else:
            raise ValueError(f"Unsupported transformation type: {transformation_type}")

    def _normalize_history_conv1d_layers(self, layers_config):
        if not isinstance(layers_config, list) or len(layers_config) == 0:
            raise ValueError(
                "history_conv1d_layers must be a non-empty list of layer configs when history_conv1d_enabled=true."
            )
        normalized = []
        for idx, layer in enumerate(layers_config):
            if not isinstance(layer, dict):
                raise ValueError(f"history_conv1d_layers[{idx}] must be an object.")
            if "filters" not in layer or "kernel_size" not in layer:
                raise ValueError(
                    f"history_conv1d_layers[{idx}] must include 'filters' and 'kernel_size'."
                )
            filters = int(layer["filters"])
            kernel_size = int(layer["kernel_size"])
            dilation_rate = int(layer.get("dilation_rate", 1))
            activation = str(layer.get("activation", "relu")).strip()
            dropout = float(layer.get("dropout", 0.0))
            use_batch_norm = bool(layer.get("use_batch_norm", False))
            use_residual = bool(layer.get("use_residual", False))

            if filters <= 0:
                raise ValueError(f"history_conv1d_layers[{idx}].filters must be > 0.")
            if kernel_size <= 0:
                raise ValueError(f"history_conv1d_layers[{idx}].kernel_size must be > 0.")
            if dilation_rate <= 0:
                raise ValueError(f"history_conv1d_layers[{idx}].dilation_rate must be > 0.")
            if not (0.0 <= dropout < 1.0):
                raise ValueError(f"history_conv1d_layers[{idx}].dropout must satisfy 0 <= dropout < 1.")
            if not activation:
                raise ValueError(f"history_conv1d_layers[{idx}].activation cannot be empty.")

            normalized.append(
                {
                    "filters": filters,
                    "kernel_size": kernel_size,
                    "dilation_rate": dilation_rate,
                    "activation": activation,
                    "dropout": dropout,
                    "use_batch_norm": use_batch_norm,
                    "use_residual": use_residual,
                }
            )
        return normalized

    def configure_history_conv1d_encoder(
        self,
        enabled=False,
        layers_config=None,
        pooling="global_max",
    ):
        """
        Optional Conv1D encoder for seen history features.

        When enabled, history features are encoded before concatenation with
        base input features. Supports:
        - rank-2 history: (batch, context_length)
        - rank-3 history: (batch, timesteps, context_length)
        """
        enabled = bool(enabled)
        self.history_conv1d_enabled = enabled
        pooling = str(pooling).strip().lower()
        if pooling not in {"global_max", "global_avg"}:
            raise ValueError("history_conv1d_pooling must be 'global_max' or 'global_avg'.")

        self.history_conv1d_encoder = None
        normalized_layers = []
        if not enabled:
            self.history_conv1d_config = {
                "enabled": False,
                "pooling": pooling,
                "layers": [],
            }
            return

        normalized_layers = self._normalize_history_conv1d_layers(layers_config)

        inp = tf.keras.Input(shape=(None, 1))
        x = inp
        for layer_cfg in normalized_layers:
            residual = x
            x = tf.keras.layers.Conv1D(
                filters=int(layer_cfg["filters"]),
                kernel_size=int(layer_cfg["kernel_size"]),
                dilation_rate=int(layer_cfg["dilation_rate"]),
                padding="causal",
                activation=None,
            )(x)
            if bool(layer_cfg["use_batch_norm"]):
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(str(layer_cfg["activation"]))(x)
            if float(layer_cfg["dropout"]) > 0.0:
                x = tf.keras.layers.Dropout(float(layer_cfg["dropout"]))(x)
            if bool(layer_cfg["use_residual"]):
                res_channels = residual.shape[-1]
                target_channels = int(layer_cfg["filters"])
                if res_channels is None or int(res_channels) != target_channels:
                    residual = tf.keras.layers.Conv1D(
                        filters=target_channels,
                        kernel_size=1,
                        padding="same",
                        activation=None,
                    )(residual)
                x = tf.keras.layers.Add()([x, residual])

        if pooling == "global_avg":
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        else:
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        self.history_conv1d_encoder = tf.keras.Model(inputs=inp, outputs=x)
        self.history_conv1d_config = {
            "enabled": True,
            "pooling": pooling,
            "layers": normalized_layers,
        }

    def _history_conv_sidecar_path(self, model_path):
        return f"{model_path}.history_conv.pkl"

    def _build_history_conv_encoder_if_needed(self, context_length_hint=1):
        encoder = getattr(self, "history_conv1d_encoder", None)
        if encoder is None:
            return
        if encoder.built:
            return
        try:
            length = int(context_length_hint)
        except Exception:
            static_len = tf.get_static_value(context_length_hint)
            length = int(static_len) if static_len is not None else 1
        length = int(max(length, 1))
        dummy = tf.zeros((1, length, 1), dtype=tf.float32)
        _ = encoder(dummy, training=False)

    def _encode_history_features(self, history_features):
        history_features = tf.convert_to_tensor(history_features, dtype=tf.float32)
        if not bool(getattr(self, "history_conv1d_enabled", False)):
            return history_features

        encoder = getattr(self, "history_conv1d_encoder", None)
        if encoder is None:
            raise ValueError(
                "history_conv1d_enabled=True but history_conv1d_encoder is not configured."
            )

        rank = len(history_features.shape)
        if rank == 2:
            # (batch, context_length) -> (batch, filters)
            x = tf.expand_dims(history_features, axis=-1)
            context_hint = history_features.shape[-1]
            self._build_history_conv_encoder_if_needed(
                context_length_hint=1 if context_hint is None else int(context_hint)
            )
            return encoder(x, training=False)
        if rank == 3:
            # (batch, timesteps, context_length) -> (batch, timesteps, filters)
            b = tf.shape(history_features)[0]
            t = tf.shape(history_features)[1]
            c = tf.shape(history_features)[2]
            flat = tf.reshape(history_features, (-1, c))
            x = tf.expand_dims(flat, axis=-1)
            context_hint = history_features.shape[-1]
            self._build_history_conv_encoder_if_needed(
                context_length_hint=1 if context_hint is None else int(context_hint)
            )
            encoded = encoder(x, training=False)
            f = tf.shape(encoded)[-1]
            return tf.reshape(encoded, (b, t, f))
        raise ValueError(
            f"Unsupported history_features rank={rank}. Expected rank 2 or 3."
        )

    def transform_input(self, instrument_paths, T_minus_t, history_features=None):
        """
        Transforms the input by concatenating instrument paths and time to maturity.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
                                        Shape: (batch_size, input_shape)
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.
                                Shape: (batch_size,)

        Returns:
        - input_data (tf.Tensor): The transformed input data.
                                Shape: (batch_size, input_shape + 1)
        """
        #instrument_paths = tf.expand_dims(instrument_paths, axis=-1)  # Shape: (n_instruments, batch_size, n_timesteps, 1)
        #T_minus_t = tf.expand_dims(T_minus_t, axis=-1)  # Shape: (batch_size, n_timesteps, 1)

        instrument_paths = self.transform_paths(instrument_paths, self.path_transformation_configs) # Shape: (batch_size, n_timesteps, n_instruments) or (batch_size, n_instruments)
        # T_minus_t  # Shape: (batch_size, n_timesteps)
        T_minus_t_expanded = tf.expand_dims(T_minus_t, axis=-1)  # Shape: (batch_size, n_timesteps, 1) or (batch_size, 1)

        # Concatenate base features [instrument paths, time-to-maturity].
        input_data = tf.concat([instrument_paths, T_minus_t_expanded], axis=-1) # Shape (batch_size, n_timesteps, n_instruments+1) or (batch_size, n_instruments+1)

        if history_features is not None:
            history_features = self._encode_history_features(history_features)
            if len(history_features.shape) != len(input_data.shape):
                raise ValueError(
                    "history_features rank mismatch: "
                    f"expected {len(input_data.shape)}, got {len(history_features.shape)}."
                )
            input_data = tf.concat([input_data, history_features], axis=-1)

        # Optional constant feature: log-strike, repeated along batch/time axes.
        if bool(getattr(self, "append_log_strike_feature", False)):
            log_k = float(getattr(self, "log_strike_value", 0.0))
            if len(input_data.shape) == 3:
                strike_feat = tf.fill(
                    [tf.shape(input_data)[0], tf.shape(input_data)[1], 1],
                    tf.constant(log_k, dtype=tf.float32),
                )
            elif len(input_data.shape) == 2:
                strike_feat = tf.fill(
                    [tf.shape(input_data)[0], 1],
                    tf.constant(log_k, dtype=tf.float32),
                )
            else:
                raise ValueError(
                    f"Unsupported input_data rank for log-strike feature: {len(input_data.shape)}"
                )
            input_data = tf.concat([input_data, strike_feat], axis=-1)

        return input_data
    
    def act(self, instrument_paths, T_minus_t, history_features=None):
        """
        Act based on the input.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The action chosen by the model.
        """
        input_data = self.transform_input(instrument_paths, T_minus_t, history_features=history_features)
        action = self.model(input_data)
        return action

    def train_batch(
        self,
        batch_paths,
        batch_T_minus_t,
        optimizer,
        loss_function,
        batch_history_features=None,
        batch_pre_history_prices=None,
    ):
        """
        Train the model on a batch of data, processing timestep by timestep.

        Arguments:
        - batch_paths (tf.Tensor): Tensor containing a batch of instrument paths. Shape: (batch_size, timesteps, input_shape)
        - batch_T_minus_t (tf.Tensor): Tensor containing the time to maturity at each timestep.
        - optimizer (tf.optimizers.Optimizer): The optimizer to use for training.

        Returns:
        - loss (tf.Tensor): The loss value after training on the batch.
        """
        with tf.GradientTape() as tape:
            try:
                params = inspect.signature(self.process_batch).parameters
            except (TypeError, ValueError):
                params = {}
            kwargs = {}
            if batch_history_features is not None:
                if "batch_history_features" in params:
                    kwargs["batch_history_features"] = batch_history_features
                elif "history_features" in params:
                    kwargs["history_features"] = batch_history_features
            if batch_pre_history_prices is not None:
                if "batch_pre_history_prices" in params:
                    kwargs["batch_pre_history_prices"] = batch_pre_history_prices
                elif "pre_history_prices" in params:
                    kwargs["pre_history_prices"] = batch_pre_history_prices
            if kwargs:
                actions = self.process_batch(batch_paths, batch_T_minus_t, **kwargs)
            else:
                actions = self.process_batch(batch_paths, batch_T_minus_t)
            loss = loss_function(batch_paths, actions)

            # Compute gradients based on the total loss
            trainable_vars = list(self.model.trainable_variables)
            if bool(getattr(self, "history_conv1d_enabled", False)):
                encoder = getattr(self, "history_conv1d_encoder", None)
                if encoder is None:
                    raise ValueError(
                        "history_conv1d_enabled=True but history_conv1d_encoder is not configured."
                    )
                trainable_vars.extend(list(encoder.trainable_variables))
            grads = tape.gradient(loss, trainable_vars)
            grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
            if grads_and_vars:
                optimizer.apply_gradients(grads_and_vars)

        return loss

    def load_model(self, model_path):
        """
        Load the model from the specified path.

        Arguments:
        - model_path (str): File path from which the model will be loaded.
        """
        if not os.path.exists(model_path):
            warnings.warn(f"Model path '{model_path}' does not exist. Exiting the load function.")
            return

        self.model = tf.keras.models.load_model(model_path)
        sidecar_path = self._history_conv_sidecar_path(model_path)
        if os.path.exists(sidecar_path):
            with open(sidecar_path, "rb") as f:
                payload = pickle.load(f)
            cfg = dict(payload.get("config", {}))
            self.configure_history_conv1d_encoder(
                enabled=bool(payload.get("enabled", False)),
                layers_config=cfg.get("layers"),
                pooling=str(cfg.get("pooling", "global_max")),
            )
            if bool(getattr(self, "history_conv1d_enabled", False)):
                weights = payload.get("weights", [])
                context_hint = int(getattr(self, "history_feature_dim", 1))
                self._build_history_conv_encoder_if_needed(context_length_hint=max(context_hint, 1))
                if weights:
                    self.history_conv1d_encoder.set_weights(weights)
        else:
            if bool(getattr(self, "history_conv1d_enabled", False)):
                raise FileNotFoundError(
                    "Expected history conv sidecar for a config with history_conv1d_enabled=true "
                    f"but not found: {sidecar_path}"
                )

    def save_model(self, model_path):
        """
        Save the model to the specified path.

        Arguments:
        - model_path (str): File path where the model will be saved.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        sidecar_path = self._history_conv_sidecar_path(model_path)
        if bool(getattr(self, "history_conv1d_enabled", False)):
            encoder = getattr(self, "history_conv1d_encoder", None)
            if encoder is None:
                raise ValueError(
                    "history_conv1d_enabled=True but history_conv1d_encoder is not configured."
                )
            context_hint = int(getattr(self, "history_feature_dim", 1))
            self._build_history_conv_encoder_if_needed(context_length_hint=max(context_hint, 1))
            payload = {
                "enabled": True,
                "config": dict(getattr(self, "history_conv1d_config", {})),
                "weights": [w.numpy() for w in encoder.weights],
            }
            with open(sidecar_path, "wb") as f:
                pickle.dump(payload, f)
        else:
            if os.path.exists(sidecar_path):
                os.remove(sidecar_path)
