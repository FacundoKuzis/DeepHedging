import tensorflow as tf
from DeepHedging.Agents import SimpleAgent

class RecurrentAgent(SimpleAgent):
    """
    A recurrent agent that processes inputs timestep by timestep and includes the accumulated position.

    Arguments:
    - input_shape (tuple): Shape of the input.
    - output_shape (int): Shape of the output.
    """
    plot_color = 'steelblue' 
    name = 'recurrent'
    is_trainable = True
    plot_name = {
        'en': 'Recurrent Agent',
        'es': 'Agente Recurrente'
    }

    def __init__(
        self,
        path_transformation_configs = None,
        n_instruments = 1,
        n_hedging_timesteps = None,
        history_feature_dim = 0,
        context_as_timesteps = True,
        context_pre_ttm_mode = "calculated",
        sequence_output_mode = "trade",
        position_activation = "linear",
        dense_units = 64,
    ):
        self.dense_units = int(dense_units)
        self.history_feature_dim = int(history_feature_dim)
        self.input_shape = (n_instruments + 1 + n_instruments + self.history_feature_dim,) # +1 for T-t and + n_instruments for accumulated position
        self.n_instruments = n_instruments
        self.context_as_timesteps = bool(context_as_timesteps)
        mode = str(context_pre_ttm_mode).strip().lower()
        if mode == "extended":
            mode = "calculated"
        if mode not in {"calculated", "zero"}:
            raise ValueError("context_pre_ttm_mode must be 'calculated' or 'zero'.")
        self.context_pre_ttm_mode = mode
        out_mode = str(sequence_output_mode).strip().lower()
        if out_mode not in {"trade", "position"}:
            raise ValueError("sequence_output_mode must be 'trade' or 'position'.")
        self.sequence_output_mode = out_mode
        pos_act = str(position_activation).strip().lower()
        if pos_act not in {"linear", "sigmoid", "tanh"}:
            raise ValueError("position_activation must be one of {'linear','sigmoid','tanh'}.")
        self.position_activation = pos_act
        self.model = self.build_model(self.input_shape, self.n_instruments)
        self.accumulated_position = None  # Initialize accumulated position
        self.path_transformation_configs = path_transformation_configs

    def build_model(self, input_shape, output_shape):
        """
        Builds a feedforward model with additional input for accumulated position.

        Arguments:
        - input_shape (tuple): Shape of the input.
        - output_shape (int): Shape of the output.

        Returns:
        - model (tf.keras.Model): The built model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),  # Adjust input shape for accumulated position and T_minus_t
            tf.keras.layers.Dense(self.dense_units, activation='relu'),
            tf.keras.layers.Dense(self.dense_units, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='linear')
        ])
        return model

    def reset_accumulated_position(self, batch_size):
        """
        Resets the accumulated position to zero for each batch.

        Arguments:
        - batch_size (int): The size of the batch being processed.
        """
        self.accumulated_position = tf.zeros((batch_size, self.n_instruments), dtype=tf.float32)

    def transform_input(self, instrument_paths, T_minus_t, history_features=None):
        """
        Transforms the input by including the accumulated position.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - transformed_input (tf.Tensor): The transformed input, including accumulated position and T_minus_t.
        """
        # Concatenate the instrument paths, accumulated position, and T_minus_t
        input_data = super().transform_input(
            instrument_paths,
            T_minus_t,
            history_features=history_features,
        )
        transformed_input = tf.concat([input_data, self.accumulated_position], axis=-1)

        return transformed_input # (batch_size, n_instruments + 1 + n_instruments)

    def _convert_model_output_to_trade_action(self, model_output):
        """
        Convert model output into trading action increments.

        - trade mode: raw model output is interpreted as action.
        - position mode: model output is interpreted as target position;
          action is target_position - current_accumulated_position.
        """
        raw = tf.convert_to_tensor(model_output, dtype=tf.float32)
        if self.sequence_output_mode == "trade":
            return raw

        if self.position_activation == "sigmoid":
            target_position = tf.sigmoid(raw)
        elif self.position_activation == "tanh":
            target_position = tf.tanh(raw)
        else:
            target_position = raw

        return target_position - self.accumulated_position

    def act(self, instrument_paths, T_minus_t, history_features=None):
        """
        Act based on the input.

        Arguments:
        - instrument_paths (tf.Tensor): Tensor containing the instrument paths at the current timestep.
        - T_minus_t (tf.Tensor): Tensor representing the time to maturity at the current timestep.

        Returns:
        - action (tf.Tensor): The action chosen by the model.
        """
        input_data = self.transform_input(
            instrument_paths,
            T_minus_t,
            history_features=history_features,
        )
        raw_output = self.model(input_data)
        action = self._convert_model_output_to_trade_action(raw_output)

        # Update accumulated position by summing actions
        self.accumulated_position += action  # Accumulate positions across timesteps

        return action

    def _run_sequence(self, seq_paths, seq_ttm, seq_history_features=None):
        all_actions = []
        n_steps = int(seq_paths.shape[1])
        for t in range(n_steps):
            current_paths = seq_paths[:, t, :]
            current_ttm = seq_ttm[:, t]
            current_history = None
            if seq_history_features is not None:
                if len(seq_history_features.shape) == 3:
                    current_history = seq_history_features[:, t, :]
                elif len(seq_history_features.shape) == 2:
                    current_history = seq_history_features
                else:
                    raise ValueError(
                        "batch_history_features must be rank-2 or rank-3 for RecurrentAgent. "
                        f"Got shape={seq_history_features.shape}."
                    )
            action = self.act(
                current_paths,
                current_ttm,
                history_features=current_history,
            )
            all_actions.append(action)
        if len(all_actions) == 0:
            return tf.zeros((seq_paths.shape[0], 0, self.n_instruments), dtype=tf.float32)
        return tf.stack(all_actions, axis=1)

    def process_batch(
        self,
        batch_paths,
        batch_T_minus_t,
        batch_history_features=None,
        batch_pre_history_prices=None,
    ):
        """
        Processes the entire batch timestep by timestep, updating the accumulated position at each step.

        Arguments:
        - batch_paths (tf.Tensor): Tensor containing a batch of instrument paths. Shape: (batch_size, timesteps, input_shape)
        - batch_T_minus_t (tf.Tensor): Tensor containing the time to maturity at each timestep.

        Returns:
        - all_actions (tf.Tensor): Tensor containing all the actions taken for the batch.
        """
        # Reset accumulated position at the start of processing each batch
        self.reset_accumulated_position(batch_paths.shape[0])
        core_paths = batch_paths[:, :-1, :]
        core_ttm = batch_T_minus_t
        history_features = batch_history_features

        if self.context_as_timesteps and batch_pre_history_prices is not None:
            pre = tf.convert_to_tensor(batch_pre_history_prices, dtype=tf.float32)
            if len(pre.shape) == 2:
                if int(self.n_instruments) != 1:
                    raise ValueError(
                        "batch_pre_history_prices rank-2 is only supported when n_instruments=1."
                    )
                pre = tf.expand_dims(pre, axis=-1)
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
                seq_actions = self._run_sequence(seq_paths, seq_ttm, seq_history_features=None)
                all_actions = seq_actions[:, -tf.shape(core_paths)[1] :, :]
            else:
                all_actions = self._run_sequence(core_paths, core_ttm, seq_history_features=history_features)
        else:
            all_actions = self._run_sequence(core_paths, core_ttm, seq_history_features=history_features)

        zero_action = tf.zeros((batch_paths.shape[0], 1, all_actions.shape[-1]))
        all_actions = tf.concat([all_actions, zero_action], axis=1)
        return all_actions
