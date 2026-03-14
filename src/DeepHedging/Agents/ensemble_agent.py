import tensorflow as tf


class EnsembleAgent:
    """
    Ensemble of multiple agents. Averages their actions at inference time.

    This agent is not trainable — it wraps pre-trained sub-agents and
    averages their process_batch outputs element-wise.
    """
    name = 'ensemble'
    is_trainable = False
    plot_color = 'darkgreen'
    plot_name = {
        'en': 'Ensemble Agent',
        'es': 'Agente Ensemble'
    }

    def __init__(self, sub_agents):
        if not sub_agents:
            raise ValueError("EnsembleAgent requires at least one sub-agent.")
        self.sub_agents = list(sub_agents)
        first = self.sub_agents[0]
        self.n_instruments = first.n_instruments
        self.path_transformation_configs = first.path_transformation_configs
        self.model = first.model
        self.context_as_timesteps = getattr(first, "context_as_timesteps", False)
        self.append_log_strike_feature = getattr(first, "append_log_strike_feature", False)
        self.log_strike_value = getattr(first, "log_strike_value", 0.0)
        if hasattr(first, "history_conv1d_enabled"):
            self.history_conv1d_enabled = first.history_conv1d_enabled
        if hasattr(first, "history_conv1d_encoder"):
            self.history_conv1d_encoder = first.history_conv1d_encoder

    def process_batch(
        self,
        batch_paths,
        batch_T_minus_t,
        batch_history_features=None,
        batch_pre_history_prices=None,
    ):
        all_actions = []
        for agent in self.sub_agents:
            actions = agent.process_batch(
                batch_paths,
                batch_T_minus_t,
                batch_history_features=batch_history_features,
                batch_pre_history_prices=batch_pre_history_prices,
            )
            all_actions.append(actions)
        return tf.reduce_mean(tf.stack(all_actions, axis=0), axis=0)
