import tensorflow as tf
from DeepHedging.Agents import LSTMAgent

class WaveNetAgent(LSTMAgent):
    """
    A WaveNet agent that processes the entire sequence of inputs using a causal WaveNet-like architecture.

    Arguments:
    - n_hedging_timesteps (int): Number of timesteps in the hedging sequence.
    - path_transformation_type (str or None): Type of transformation applied to the instrument paths.
    - K (float or None): Strike price or other parameter for path transformation.
    - num_filters (int): Number of filters in the convolutional layers.
    - num_residual_blocks (int): Number of residual blocks in the WaveNet model.
    """

    plot_color = 'pink' 
    name = 'wavenet'
    is_trainable = True
    plot_name = {
        'en': 'WaveNet Agent',
        'es': 'Agente WaveNet'
    }
    
    def __init__(
        self,
        n_hedging_timesteps,
        path_transformation_configs=None,
        num_filters=32,
        num_residual_blocks=3,
        wavenet_block_configs=None,
        wavenet_activation="relu",
        wavenet_use_skip_connections=True,
        wavenet_output_hidden_filters=0,
        n_instruments=1,
        history_feature_dim=0,
        context_as_timesteps=True,
        context_pre_ttm_mode="calculated",
        sequence_output_mode="trade",
        position_activation="linear",
    ):
        
        self.history_feature_dim = int(history_feature_dim)
        # Keep time dimension dynamic so context can be consumed as sequence prefix.
        self.input_shape = (None, n_instruments + 1 + self.history_feature_dim) # +1 for T-t
        self.n_instruments = n_instruments
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.wavenet_activation = str(wavenet_activation).strip().lower()
        self.wavenet_use_skip_connections = bool(wavenet_use_skip_connections)
        self.wavenet_output_hidden_filters = int(wavenet_output_hidden_filters)
        self.wavenet_block_configs = self._normalize_block_configs(wavenet_block_configs)
        self.context_as_timesteps = bool(context_as_timesteps)
        out_mode = str(sequence_output_mode).strip().lower()
        if out_mode not in {"trade", "position"}:
            raise ValueError("sequence_output_mode must be 'trade' or 'position'.")
        self.sequence_output_mode = out_mode
        pos_act = str(position_activation).strip().lower()
        if pos_act not in {"linear", "sigmoid", "tanh"}:
            raise ValueError("position_activation must be one of {'linear','sigmoid','tanh'}.")
        self.position_activation = pos_act
        mode = str(context_pre_ttm_mode).strip().lower()
        if mode == "extended":
            mode = "calculated"
        if mode not in {"calculated", "zero"}:
            raise ValueError("context_pre_ttm_mode must be 'calculated' or 'zero'.")
        self.context_pre_ttm_mode = mode
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.path_transformation_configs = path_transformation_configs

    def _normalize_block_configs(self, wavenet_block_configs):
        if wavenet_block_configs is None:
            return [
                {
                    "filters": int(self.num_filters),
                    "kernel_size": 2,
                    "dilation_rate": int(2 ** i),
                    "activation": str(self.wavenet_activation),
                    "dropout": 0.0,
                    "use_batch_norm": False,
                    "use_residual": True,
                    "use_skip": True,
                    "gated": False,
                }
                for i in range(int(self.num_residual_blocks))
            ]

        if not isinstance(wavenet_block_configs, list) or len(wavenet_block_configs) == 0:
            raise ValueError("wavenet_block_configs must be a non-empty list when provided.")

        normalized = []
        for idx, cfg in enumerate(wavenet_block_configs):
            if not isinstance(cfg, dict):
                raise ValueError(f"wavenet_block_configs[{idx}] must be an object.")
            if "filters" not in cfg or "kernel_size" not in cfg:
                raise ValueError(
                    f"wavenet_block_configs[{idx}] must include 'filters' and 'kernel_size'."
                )
            filters = int(cfg["filters"])
            kernel_size = int(cfg["kernel_size"])
            dilation_rate = int(cfg.get("dilation_rate", 1))
            if filters <= 0:
                raise ValueError(f"wavenet_block_configs[{idx}].filters must be > 0.")
            if kernel_size <= 0:
                raise ValueError(f"wavenet_block_configs[{idx}].kernel_size must be > 0.")
            if dilation_rate <= 0:
                raise ValueError(f"wavenet_block_configs[{idx}].dilation_rate must be > 0.")
            dropout = float(cfg.get("dropout", 0.0))
            if dropout < 0.0 or dropout >= 1.0:
                raise ValueError(
                    f"wavenet_block_configs[{idx}].dropout must satisfy 0 <= dropout < 1."
                )

            activation = str(cfg.get("activation", self.wavenet_activation)).strip().lower()
            if not activation:
                raise ValueError(f"wavenet_block_configs[{idx}].activation cannot be empty.")

            normalized.append(
                {
                    "filters": filters,
                    "kernel_size": kernel_size,
                    "dilation_rate": dilation_rate,
                    "activation": activation,
                    "dropout": dropout,
                    "use_batch_norm": bool(cfg.get("use_batch_norm", False)),
                    "use_residual": bool(cfg.get("use_residual", True)),
                    "use_skip": bool(cfg.get("use_skip", True)),
                    "gated": bool(cfg.get("gated", False)),
                }
            )
        return normalized

    @staticmethod
    def _apply_activation(x, activation: str):
        act = str(activation).strip().lower()
        if act in {"linear", "identity", "none"}:
            return x
        return tf.keras.layers.Activation(act)(x)

    def build_model(self, input_shape, output_shape):
        """
        Builds a causal WaveNet-like model.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        # Initial causal convolutional layer
        x = tf.keras.layers.Conv1D(
            filters=int(self.wavenet_block_configs[0]["filters"]),
            kernel_size=1,
            padding="causal",
            activation="linear",
        )(x)
        x = self._apply_activation(x, self.wavenet_activation)

        skip_connections = []
        skip_filters = int(self.wavenet_block_configs[0]["filters"])

        for block_cfg in self.wavenet_block_configs:
            residual = x
            filters = int(block_cfg["filters"])
            kernel_size = int(block_cfg["kernel_size"])
            dilation_rate = int(block_cfg["dilation_rate"])
            activation = str(block_cfg["activation"])

            if bool(block_cfg["gated"]):
                h = tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    activation="tanh",
                )(x)
                g = tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    activation="sigmoid",
                )(x)
                x = tf.keras.layers.Multiply()([h, g])
            else:
                x = tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    activation="linear",
                )(x)
                x = self._apply_activation(x, activation)

            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                padding="causal",
                activation="linear",
            )(x)
            x = self._apply_activation(x, activation)

            if bool(block_cfg["use_batch_norm"]):
                x = tf.keras.layers.BatchNormalization()(x)
            if float(block_cfg["dropout"]) > 0.0:
                x = tf.keras.layers.Dropout(float(block_cfg["dropout"]))(x)

            if self.wavenet_use_skip_connections and bool(block_cfg["use_skip"]):
                skip_connections.append(
                    tf.keras.layers.Conv1D(
                        filters=skip_filters,
                        kernel_size=1,
                        padding="causal",
                        activation="linear",
                    )(x)
                )

            if bool(block_cfg["use_residual"]):
                residual_channels = residual.shape[-1]
                if residual_channels is None or int(residual_channels) != filters:
                    residual = tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=1,
                        padding="causal",
                        activation="linear",
                    )(residual)
                x = tf.keras.layers.Add()([x, residual])

        if self.wavenet_use_skip_connections and len(skip_connections) > 0:
            if len(skip_connections) == 1:
                x = skip_connections[0]
            else:
                x = tf.keras.layers.Add()(skip_connections)

        if self.wavenet_output_hidden_filters > 0:
            x = tf.keras.layers.Conv1D(
                filters=int(self.wavenet_output_hidden_filters),
                kernel_size=1,
                padding="causal",
                activation="linear",
            )(x)
            x = self._apply_activation(x, self.wavenet_activation)

        # Final convolutional layer to produce the output
        x = tf.keras.layers.Conv1D(
            filters=output_shape,
            kernel_size=1,
            padding='causal',
            activation='linear'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model
