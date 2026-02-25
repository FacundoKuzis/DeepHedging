import tensorflow as tf
from DeepHedging.Agents import BaseAgent

class LSTMAgent(BaseAgent):
    """
    An LSTM agent that processes the entire sequence of inputs at once.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """

    plot_color = 'forestgreen' 
    name = 'lstm'
    is_trainable = True
    plot_name = {
        'en': 'LSTM Agent',
        'es': 'Agente LSTM'
    }

    def __init__(
        self,
        n_hedging_timesteps,
        path_transformation_configs=None,
        n_instruments=1,
        history_feature_dim=0,
        context_as_timesteps=True,
        context_pre_ttm_mode="calculated",
        sequence_output_mode="trade",
        position_activation="linear",
    ):

        self.history_feature_dim = int(history_feature_dim)
        # Keep time dimension dynamic so sequence models can consume optional
        # pre-history as temporal prefix: (L + N) timesteps instead of only N.
        self.input_shape = (None, n_instruments + 1 + self.history_feature_dim) # +1 for T-t
        self.n_instruments = n_instruments
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

    def build_model(self, input_shape, output_shape):
        """
        Builds an LSTM model.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LSTM(30, return_sequences=True),
            tf.keras.layers.LSTM(30, return_sequences=True),
            tf.keras.layers.Dense(output_shape, activation='linear')
        ])
        return model

    def _convert_sequence_output_to_actions(self, model_output):
        """
        Convert sequence model outputs into trading actions.

        - trade mode: output is interpreted directly as action increments.
        - position mode: output is interpreted as target position, then actions
          are first differences of positions.
        """
        x = tf.convert_to_tensor(model_output, dtype=tf.float32)
        if self.sequence_output_mode == "trade":
            return x

        if self.position_activation == "sigmoid":
            positions = tf.sigmoid(x)
        elif self.position_activation == "tanh":
            positions = tf.tanh(x)
        else:
            positions = x

        prev_positions = tf.concat(
            [tf.zeros_like(positions[:, :1, :]), positions[:, :-1, :]],
            axis=1,
        )
        return positions - prev_positions

    def process_batch(
        self,
        batch_paths,
        batch_T_minus_t,
        batch_history_features=None,
        batch_pre_history_prices=None,
    ):
        core_paths = batch_paths[:, :-1, :]  # (batch_size, N, n_instruments)
        core_ttm = batch_T_minus_t  # (batch_size, N)
        history_features = batch_history_features

        if self.context_as_timesteps and batch_pre_history_prices is not None:
            pre = tf.convert_to_tensor(batch_pre_history_prices, dtype=tf.float32)
            if len(pre.shape) == 2:
                if int(self.n_instruments) != 1:
                    raise ValueError(
                        "batch_pre_history_prices rank-2 is only supported when n_instruments=1."
                    )
                pre = tf.expand_dims(pre, axis=-1)  # (batch_size, L, 1)
            elif len(pre.shape) != 3:
                raise ValueError(
                    "batch_pre_history_prices must be rank-2 or rank-3. "
                    f"Got shape={pre.shape}."
                )
            if int(pre.shape[2]) != int(self.n_instruments):
                raise ValueError(
                    "batch_pre_history_prices instrument dimension mismatch: "
                    f"expected {self.n_instruments}, got {int(pre.shape[2])}."
                )
            if int(pre.shape[1]) > 0:
                if self.context_pre_ttm_mode == "zero":
                    pre_ttm = tf.zeros(
                        (tf.shape(pre)[0], tf.shape(pre)[1]),
                        dtype=core_ttm.dtype,
                    )
                else:
                    # Estimated from the hedge-grid: dt = (T_t0 - T_t1).
                    # Pre-history at t=-k gets T_minus_t = T_t0 + k*dt.
                    n_core_steps = tf.shape(core_ttm)[1]
                    base_ttm = core_ttm[:, :1]
                    dt = tf.where(
                        n_core_steps > 1,
                        core_ttm[:, :1] - core_ttm[:, 1:2],
                        base_ttm,
                    )
                    l = tf.shape(pre)[1]
                    offsets = tf.cast(
                        tf.range(l, 0, -1),
                        dtype=core_ttm.dtype,
                    )
                    pre_ttm = base_ttm + offsets[tf.newaxis, :] * dt
                seq_paths = tf.concat([pre, core_paths], axis=1)
                seq_ttm = tf.concat([pre_ttm, core_ttm], axis=1)
                # In temporal-prefix mode, context is injected through time axis.
                history_features = None
                seq_actions = self.act(
                    seq_paths,
                    seq_ttm,
                    history_features=history_features,
                )  # (batch_size, L+N, n_instruments)
                seq_actions = self._convert_sequence_output_to_actions(seq_actions)
                all_actions = seq_actions[:, -tf.shape(core_paths)[1] :, :]
            else:
                all_actions = self.act(
                    core_paths,
                    core_ttm,
                    history_features=history_features,
                )
                all_actions = self._convert_sequence_output_to_actions(all_actions)
        else:
            all_actions = self.act(
                core_paths,
                core_ttm,
                history_features=history_features,
            ) # (batch_size, N, n_instruments)
            all_actions = self._convert_sequence_output_to_actions(all_actions)

        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[2]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)
        #all_actions = all_actions[:, :, 0] # temporary, only one instrument

        return all_actions

