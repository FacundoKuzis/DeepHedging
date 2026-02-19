import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings
import pandas as pd
import random
import time
from tqdm import tqdm  # For progress bars


class Environment:
    def __init__(self, agent, T, N, r, instrument_list, n_instruments, contingent_claim, cost_function, 
                 risk_measure = None,
                 n_epochs = None, batch_size = None, learning_rate = None, optimizer = None,
                 resample_each_epoch=False, train_random_seed=None, val_random_seed=None):
        self.agent = agent
        self.T = T # Maturity (in years)
        self.N = N # Number of hedging steps
        self.dt = self.T / self.N # Time increment
        self.r = r # Risk Free yearly rate
        self.instrument_list = instrument_list
        self.n_instruments = n_instruments # It may not be the same as len of instrument_list (e.g. HestonStock with return_variance)
        self.contingent_claim = contingent_claim
        self.cost_function = cost_function
        self.risk_measure = risk_measure
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = None if not optimizer else optimizer(learning_rate=learning_rate)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.train_random_seed = train_random_seed
        self.val_random_seed = val_random_seed

        self.train_losses = []
        self.val_losses = []

        if self.risk_measure is None:
            warnings.warn(
                "Environment created without risk_measure. "
                "Training/testing will fail until a valid risk measure is provided."
            )

    def _derive_seed(self, base_seed, offset):
        if base_seed is None:
            return None
        return int(base_seed) + int(offset)

    def _select_claim_paths(self, paths):
        if hasattr(self.contingent_claim, "select_underlying_paths"):
            return self.contingent_claim.select_underlying_paths(paths)

        # Backward-compat fallback.
        return paths[:, :, 0]

    def generate_data(self, n_paths, random_seed=None):

        data = tf.TensorArray(dtype=tf.float32, size=self.n_instruments)
        
        i = 0
        for instrument_idx, instrument in enumerate(self.instrument_list):
            instrument_seed = self._derive_seed(random_seed, instrument_idx)
            instrument_data = instrument.generate_paths(n_paths, random_seed=instrument_seed)
            if isinstance(instrument_data, tuple):
                for individual_data in instrument_data:
                    data = data.write(i, individual_data)
                    i += 1
            else:
                data = data.write(i, instrument_data)
                i += 1

        if i != self.n_instruments:
            raise ValueError(
                f"n_instruments mismatch: expected {self.n_instruments}, generated {i}."
            )
        
        # Stack all the instrument data into a single tensor
        data = data.stack()

        data_transposed = tf.transpose(data, perm=[1, 2, 0])
        
        return data_transposed # (n_paths, N+1, n_instruments)

    def calculate_pnl(self, paths, actions, include_decomposition = False):
        # Calculate the portfolio value at each time step
        portfolio_values = tf.cumsum(actions, axis=1) * paths # (batch_size, N+1, n_instruments)        
        final_portfolio_value = portfolio_values[:, -1, :] # (batch_size, n_instruments)
        final_portfolio_value = tf.reduce_sum(final_portfolio_value, axis = 1) # (batch_size)

        # Calculate the total transaction costs
        costs = self.cost_function(actions, paths)

        # Calculate the cash flows from purchases and transaction costs
        purchases_cashflows = -actions * paths - costs
        purchases_cashflows = tf.reduce_sum(purchases_cashflows, axis = 2)

        # Calculate the factor for compounding cash positions
        factor = (1 + self.r) ** self.dt - 1

        # Initialize the cash tensor with the first cash flow
        cash = purchases_cashflows[:, 0:1]

        # Iterate over the remaining time steps to accumulate cash values
        for t in range(1, purchases_cashflows.shape[1]):
            current_cash = cash[:, -1:] * (1 + factor) + purchases_cashflows[:, t:t+1]
            cash = tf.concat([cash, current_cash], axis=1)

        final_cash_positions = cash[:, -1] # (batch_size)

        # Calculate contingent-claim payoff using explicit underlying mapping.
        claim_paths = self._select_claim_paths(paths)
        payoff = self.contingent_claim.calculate_payoff(claim_paths)
        
        # Calculate the PnL
        pnl = final_portfolio_value + final_cash_positions - payoff
        
        if include_decomposition:
            return pnl, portfolio_values, cash, payoff
        else:
            return pnl

    def loss_function(self, paths, actions):
        if self.risk_measure is None:
            raise ValueError("risk_measure is not set in Environment.")
        pnl = self.calculate_pnl(paths, actions)
        return self.risk_measure.calculate(pnl)

    def train(
        self,
        train_paths,
        val_paths=0,
        random_seed=None,
        val_random_seed=None,
        epoch_end_callback=None,
    ):
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized. Provide optimizer and learning_rate when creating Environment.")

        train_seed = self.train_random_seed if random_seed is None else random_seed
        val_seed = self.val_random_seed if val_random_seed is None else val_random_seed
        if val_seed is None:
            val_seed = self._derive_seed(train_seed, 1_000_000)

        if train_seed is not None:
            random.seed(int(train_seed))
            np.random.seed(int(train_seed))
            tf.random.set_seed(int(train_seed))
            try:
                tf.keras.utils.set_random_seed(int(train_seed))
            except Exception:
                pass

        train_data = None
        if not self.resample_each_epoch:
            train_data = self.generate_data(train_paths, random_seed=train_seed) # (n_paths, N+1, n_instruments)

        if val_paths > 0:
            val_data = self.generate_data(val_paths, random_seed=val_seed)
            T_minus_t_val =  self.get_T_minus_t(val_paths)

        total_epochs = int(self.n_epochs)
        if total_epochs <= 0:
            raise ValueError("n_epochs must be > 0.")

        def _current_lr_value():
            lr = self.optimizer.learning_rate
            try:
                return float(tf.keras.backend.get_value(lr))
            except Exception:
                try:
                    return float(lr.numpy())
                except Exception:
                    return float(lr)

        def _set_lr_value(new_lr):
            lr = self.optimizer.learning_rate
            try:
                tf.keras.backend.set_value(lr, float(new_lr))
                return
            except Exception:
                pass
            try:
                if hasattr(lr, "assign"):
                    lr.assign(float(new_lr))
                    return
            except Exception:
                pass
            try:
                self.optimizer.learning_rate = float(new_lr)
                return
            except Exception as exc:
                raise ValueError(f"Unable to set optimizer learning_rate to {new_lr}.") from exc

        epoch = 0
        while epoch < total_epochs:
            if self.resample_each_epoch:
                epoch_seed = self._derive_seed(train_seed, epoch)
                train_data = self.generate_data(train_paths, random_seed=epoch_seed)

            # Training
            epoch_losses = []
            for i in range(0, train_paths, self.batch_size):
                batch_paths = train_data[i:i+self.batch_size]
                batch_T_minus_t = self.get_T_minus_t(batch_paths.shape[0])
                loss = self.agent.train_batch(batch_paths, batch_T_minus_t, self.optimizer, self.loss_function)
                epoch_losses.append(loss.numpy())

            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)

            # Validation
            if val_paths > 0:
                val_actions = self.agent.process_batch(val_data, T_minus_t_val)
                val_loss = self.loss_function(val_data, val_actions)
                self.val_losses.append(val_loss.numpy())
                print(
                    f"Epoch {epoch+1}/{total_epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, LR: {_current_lr_value():.6g}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{total_epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"LR: {_current_lr_value():.6g}"
                )

            if epoch_end_callback is not None:
                epoch_info = {
                    "epoch": int(epoch + 1),
                    "planned_epochs": int(total_epochs),
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(self.val_losses[-1]) if val_paths > 0 else None,
                    "learning_rate": float(_current_lr_value()),
                }
                callback_result = epoch_end_callback(epoch_info)

                stop_training = False
                if isinstance(callback_result, dict):
                    new_lr = callback_result.get("set_learning_rate")
                    if new_lr is not None:
                        _set_lr_value(new_lr)

                    extend_by = callback_result.get("extend_n_epochs_by")
                    if extend_by is not None:
                        extend_by = max(0, int(extend_by))
                        total_epochs += extend_by

                    if bool(callback_result.get("stop_training", False)):
                        stop_training = True
                elif callback_result is False:
                    stop_training = True

                if stop_training:
                    break

            epoch += 1

        if val_paths > 0:
            return self.train_losses, self.val_losses
        else:
            return self.train_losses

    def test(self, paths_to_test = None, n_paths = None, random_seed = None, 
             plot_pnl = False, plot_title = 'Distribucion de PnL', save_plot_path = None):
        
        if paths_to_test is not None:
            paths = paths_to_test
        elif n_paths is not None:
            paths = self.generate_data(n_paths, random_seed = random_seed)
        else:
            raise ValueError('Insert either paths_to_test or n_paths.')

        if not isinstance(paths, tf.Tensor):
            paths = tf.convert_to_tensor(paths, dtype=tf.float32)
        
        T_minus_t_single = tf.range(self.N, 0, -1, dtype=tf.float32) * self.dt
        T_minus_t = tf.tile(tf.expand_dims(T_minus_t_single, axis=0), [paths.shape[0], 1])

        val_actions = self.agent.process_batch(paths, T_minus_t)
        loss = self.loss_function(paths, val_actions)

        if plot_pnl:
            pnl = self.calculate_pnl(paths, val_actions)
            
            # Improved histogram plot
            plt.figure(figsize=(10, 6))
            plt.hist(pnl, bins=30, color='blue', alpha=0.7, edgecolor='black')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'{plot_title}. Perdida: {loss:.4f}', fontsize=14)
            plt.xlabel('PnL', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)

            if save_plot_path:
                plt.savefig(save_plot_path)
                print(f"Plot saved to {save_plot_path}")
            else:
                plt.show()
                
        return loss

    def get_T_minus_t(self, shape):
        T_minus_t_single = tf.range(self.N, 0, -1, dtype=tf.float32) * self.dt
        T_minus_t = tf.tile(tf.expand_dims(T_minus_t_single, axis=0), [shape, 1])
        return T_minus_t

    def _get_agent_plot_name(self, agent, language='es'):
        plot_name = getattr(agent, 'plot_name', None)
        if isinstance(plot_name, dict):
            return plot_name.get(language, agent.name)
        if isinstance(plot_name, str):
            return plot_name
        return getattr(agent, 'name', 'unknown_agent')

    def _get_agent_identifier(self, agent, idx):
        if hasattr(agent, "agent_id"):
            return str(agent.agent_id)
        return f"{idx:02d}_{getattr(agent, 'name', 'unknown_agent')}"

    def _get_agent_plot_color(self, agent, idx=0):
        color = getattr(agent, "plot_color", None)
        if isinstance(color, str) and color.strip():
            return color
        cmap = plt.cm.get_cmap("tab10", 10)
        return cmap(idx % 10)

    def terminal_hedging_error_multiple_agents(
        self,
        agents,
        n_paths=10_000,
        random_seed=None,
        plot_error=False,
        plot_title='Error de Cobertura Terminal',
        save_plot_path=None,
        colors=None,
        save_stats_path=None,
        loss_functions=None,
        min_x=-0.3,
        max_x=0.3,
        language='es',
        save_actions_path=None,
        fixed_actions_paths=None,
        pricing_method='fixed',
        agent_eval_batch_size=None,
        progress_log_every_agent_batches=5,
    ):
        """
        Computes terminal hedging error for multiple agents, generates plots, and saves statistics.

        Arguments:
        - agents (list): List of agent instances to evaluate.
        - pricing_method (str): 'fixed' to use the price of the first agent for all agents,
                                'individual' to compute and use each agent's own price.

        Returns:
        - mean_errors (list): List of mean errors for each agent.
        - std_errors (list): List of standard deviations of errors for each agent.
        - losses (dict or None): Dictionary of loss function results for each agent.
        """
        eval_start_time = time.perf_counter()

        # Generate the data paths
        paths = self.generate_data(n_paths, random_seed=random_seed)
        print(
            f"[terminal] Paths generated: shape={tuple(paths.shape)}, "
            f"n_paths={n_paths}, n_agents={len(agents)}"
        )

        # Compute prices based on the pricing_method
        if pricing_method == 'fixed':
            # Use the price of the first agent
            first_agent = agents[0]
            price = self.get_agent_price(first_agent, n_paths=n_paths, random_seed=random_seed)
            prices = [price] * len(agents)
            print(f"Using fixed price from the first agent: {price}")
        elif pricing_method == 'individual':
            # Compute the price for each agent
            prices = []
            for agent in agents:
                price = self.get_agent_price(agent, n_paths=n_paths, random_seed=random_seed)
                prices.append(price)
                print(f"Computed price for agent '{agent.name}': {price}")
        else:
            raise ValueError(f"Invalid pricing_method '{pricing_method}'. Choose 'fixed' or 'individual'.")

        errors = []
        mean_errors = []
        std_errors = []

        # Initialize a dictionary to store losses for each loss function and agent
        loss_results = {loss_fn.name: [] for loss_fn in loss_functions} if loss_functions else {}

        # Ensure save_actions_path exists if provided
        if save_actions_path:
            os.makedirs(save_actions_path, exist_ok=True)

        # If fixed_actions_paths is not provided, check for existing action files in save_actions_path
        if fixed_actions_paths is None and save_actions_path:
            fixed_actions_paths = {}
            for idx, agent in enumerate(agents):
                agent_id = self._get_agent_identifier(agent, idx)
                agent_actions_filename = f"{agent_id}_actions.npy"
                agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                if os.path.isfile(agent_actions_path):
                    fixed_actions_paths[agent_id] = agent_actions_path
                    print(f"Using existing actions file for agent '{agent_id}' at '{agent_actions_path}'.")
            # If no existing files are found, set fixed_actions_paths back to None
            if not fixed_actions_paths:
                fixed_actions_paths = None

        for idx, agent in enumerate(agents):
            agent_name = agent.name
            agent_id = self._get_agent_identifier(agent, idx)
            price = prices[idx]
            agent_start_time = time.perf_counter()
            print(f"[terminal] agent={agent_id} ({idx+1}/{len(agents)}): starting evaluation.")

            # Check if actions are fixed for this agent
            fixed_path = None
            if fixed_actions_paths:
                if agent_id in fixed_actions_paths:
                    fixed_path = fixed_actions_paths[agent_id]
                elif agent_name in fixed_actions_paths:  # backward compatibility
                    fixed_path = fixed_actions_paths[agent_name]
            if fixed_path is not None:
                if not os.path.isfile(fixed_path):
                    raise FileNotFoundError(f"Fixed actions file for agent '{agent_id}' not found at '{fixed_path}'.")
                # Load val_actions from .npy
                print(f"Loading fixed actions for agent '{agent_id}' from '{fixed_path}'.")
                val_actions_np = np.load(fixed_path)
                val_actions = tf.convert_to_tensor(val_actions_np, dtype=tf.float32)
            else:
                total_paths = int(paths.shape[0])
                if agent_eval_batch_size is None:
                    batch_size_eval = total_paths
                else:
                    batch_size_eval = max(1, int(agent_eval_batch_size))
                if batch_size_eval >= total_paths:
                    print(
                        f"[terminal] agent={agent_id}: computing actions in one batch "
                        f"(n_paths={total_paths})."
                    )
                    T_minus_t = self.get_T_minus_t(total_paths)
                    val_actions = agent.process_batch(paths, T_minus_t)
                else:
                    n_batches = int(np.ceil(total_paths / float(batch_size_eval)))
                    progress_step = max(1, int(progress_log_every_agent_batches))
                    print(
                        f"[terminal] agent={agent_id}: computing actions in {n_batches} batches "
                        f"(batch_size={batch_size_eval})."
                    )
                    action_chunks = []
                    for b_idx in range(n_batches):
                        start = b_idx * batch_size_eval
                        end = min((b_idx + 1) * batch_size_eval, total_paths)
                        batch_paths = paths[start:end]
                        batch_t_minus_t = self.get_T_minus_t(end - start)
                        action_chunks.append(agent.process_batch(batch_paths, batch_t_minus_t))

                        done = b_idx + 1
                        if done % progress_step == 0 or done == n_batches:
                            print(
                                f"[terminal] agent={agent_id}: actions batches {done}/{n_batches} "
                                f"({end}/{total_paths} paths)."
                            )
                    val_actions = tf.concat(action_chunks, axis=0)

                # Save val_actions if save_actions_path is provided
                if save_actions_path:
                    agent_actions_filename = f"{agent_id}_actions.npy"
                    agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                    # Convert TensorFlow tensor to NumPy array
                    val_actions_np = val_actions.numpy()
                    # Save as .npy
                    np.save(agent_actions_path, val_actions_np)
                    print(f"Saved val_actions for agent '{agent_id}' to '{agent_actions_path}'.")

            actions_elapsed = time.perf_counter() - agent_start_time
            print(f"[terminal] agent={agent_id}: actions ready in {actions_elapsed:.2f}s. Calculating PnL...")
            pnl = self.calculate_pnl(paths, val_actions)
            error = price + pnl * np.exp(-self.r * self.T)

            errors.append(error)
            mean_errors.append(tf.reduce_mean(error).numpy())
            std_errors.append(tf.math.reduce_std(error).numpy())

            # Compute additional loss functions if provided
            if loss_functions:
                for loss_fn in loss_functions:
                    loss_value = loss_fn(pnl)
                    loss_results[loss_fn.name].append(tf.reduce_mean(loss_value).numpy())

            total_agent_elapsed = time.perf_counter() - agent_start_time
            print(f"[terminal] agent={agent_id}: done in {total_agent_elapsed:.2f}s.")

        if plot_error:
            plt.figure(figsize=(10, 6))

            # Define the bins for the histogram
            bins = np.linspace(min_x, max_x, 60)  # 60 bins across the specified range

            # Use explicit colors if provided; otherwise prioritize each agent.plot_color.
            if colors is None:
                resolved_colors = [self._get_agent_plot_color(agent, i) for i, agent in enumerate(agents)]
            else:
                if len(colors) < len(agents):
                    raise ValueError(
                        f"colors length mismatch: expected at least {len(agents)}, got {len(colors)}."
                    )
                resolved_colors = list(colors)
                for i, agent in enumerate(agents):
                    if resolved_colors[i] is None:
                        resolved_colors[i] = self._get_agent_plot_color(agent, i)

            resolved_title = plot_title
            if language == 'es' and (plot_title is None or str(plot_title).strip() == "" or str(plot_title).strip() == "Terminal Hedging Error"):
                resolved_title = "Error de Cobertura Terminal"

            # Plot each agent's error histogram
            for i, error in enumerate(errors):
                plt.hist(error, bins=bins, density=True, color=resolved_colors[i], alpha=0.6, edgecolor='black', 
                        label=self._get_agent_plot_name(agents[i], language))

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(resolved_title, fontsize=14)
            if language == 'es':
                plt.xlabel('Error de Cobertura', fontsize=12)
            else:
                plt.xlabel('Error', fontsize=12)

            # Set y-axis label based on language
            if language == 'es':
                plt.ylabel('Densidad', fontsize=12)
            else:
                plt.ylabel('Density', fontsize=12)

            plt.legend()

            if save_plot_path:
                os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
                plt.savefig(save_plot_path)
                print(f"Plot saved to {save_plot_path}")
                plt.close()
            else:
                plt.show()

        # Save statistics to Excel if save_stats_path is provided
        if save_stats_path:
            os.makedirs(os.path.dirname(save_stats_path), exist_ok=True)
            data = {
                'Agent': [self._get_agent_plot_name(agent, language) for agent in agents],
                'Mean Error': mean_errors,
                'Standard Deviation': std_errors,
            }
            # Add additional loss functions to the data dictionary
            if loss_functions:
                for loss_fn_name, results in loss_results.items():
                    data[loss_fn_name] = results

            df = pd.DataFrame(data)
            df.to_excel(save_stats_path, index=False)
            print(f"Statistics saved to {save_stats_path}")
            total_elapsed = time.perf_counter() - eval_start_time
            print(f"[terminal] Completed multi-agent terminal evaluation in {total_elapsed:.2f}s.")
            return df

        total_elapsed = time.perf_counter() - eval_start_time
        print(f"[terminal] Completed multi-agent terminal evaluation in {total_elapsed:.2f}s.")
        return mean_errors, std_errors, loss_results if loss_functions else None

    def get_agent_price(self, agent, n_paths=10_000, random_seed=None):
        """
        Calculates the price of the agent by computing the expected loss over a set of paths.

        Arguments:
        - agent: The agent instance whose price we want to compute.
        - n_paths: Number of paths to generate.
        - random_seed: Random seed for path generation.

        Returns:
        - price: The computed price as loss * exp(-r * T)
        """
        if agent.is_trainable:
            paths = self.generate_data(n_paths, random_seed=random_seed)
            T_minus_t = self.get_T_minus_t(paths.shape[0])
            actions = agent.process_batch(paths, T_minus_t)
            pnl = self.calculate_pnl(paths, actions)
            loss = self.risk_measure.calculate(pnl)
            price = loss.numpy() * np.exp(-self.r * self.T)
        else:
            price = agent.get_model_price()
        return price

    def save_optimizer(self, optimizer_path):
        """
        Save the optimizer state and object to the specified path using optimizer.get_config() and optimizer.variables().

        Arguments:
        - optimizer_path (str): Directory path where the optimizer state will be saved.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized. Cannot save optimizer state.")

        # Ensure the directory exists
        os.makedirs(optimizer_path, exist_ok=True)

        # Save the optimizer configuration
        optimizer_config = self.optimizer.get_config()
        with open(os.path.join(optimizer_path, 'optimizer_config.pkl'), 'wb') as f:
            pickle.dump(optimizer_config, f)

        # Save the optimizer variables (weights)
        optimizer_weights = [variable.numpy() for variable in self.optimizer.variables]
        with open(os.path.join(optimizer_path, 'optimizer_weights.pkl'), 'wb') as f:
            pickle.dump(optimizer_weights, f)

    def load_optimizer(self, optimizer_path, only_weights = False):
        """
        Load the optimizer state and object from the specified path using optimizer.get_config() and optimizer.variables().

        Arguments:
        - optimizer_path (str): Directory path from which the optimizer state will be loaded.
        """
        if not os.path.exists(optimizer_path):
            warnings.warn(f"Optimizer path '{optimizer_path}' does not exist. Exiting the load function.")
            return

        if self.optimizer is None and only_weights:
            raise ValueError("Optimizer is not initialized. Initialize Environment with an optimizer before loading only weights.")

        variables_index_start = 2
        if not only_weights:
            # Load the optimizer configuration
            with open(os.path.join(optimizer_path, 'optimizer_config.pkl'), 'rb') as f:
                optimizer_config = pickle.load(f)
            
            # Re-instantiate the optimizer using the loaded configuration
            self.optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config)
            variables_index_start = 0

        # Initialize the optimizer's variables by applying it to some dummy data
        dummy_data = [tf.zeros_like(var) for var in self.agent.model.trainable_variables]
        self.optimizer.apply_gradients(zip(dummy_data, self.agent.model.trainable_variables))

        # Load the optimizer weights
        with open(os.path.join(optimizer_path, 'optimizer_weights.pkl'), 'rb') as f:
            optimizer_weights = pickle.load(f)

        # Apply the loaded weights to the optimizer
        for variable, weight in zip(self.optimizer.variables[variables_index_start:], optimizer_weights[variables_index_start:]):
            variable.assign(weight)

    def plot_hedging_strategy(self, save_plot_path=None, random_seed = None):
        """
        Generates one path, gets the hedging actions, and plots the portfolio value over time.
        Also plots the contingent claim payoff at the final timestep and prints the loss value.
        Additionally, plots the underlying asset path on a secondary axis.

        Arguments:
        - save_plot_path (str, optional): File path to save the plot. If None, the plot is only shown.
        """

        # Generate a single path
        single_path = self.generate_data(1, random_seed = random_seed)  # Shape: (1, N+1, n_instruments)
        T_minus_t_single = tf.range(self.N, 0, -1, dtype=tf.float32) * self.dt
        T_minus_t_single = tf.expand_dims(T_minus_t_single, axis=0)  # Shape: (1, N)

        # Get the hedging actions
        actions = self.agent.process_batch(single_path, T_minus_t_single)  # Shape: (1, N)

        pnl, portfolio_values, cash, payoff = self.calculate_pnl(single_path, actions, include_decomposition = True)

        net_portfolio_values = tf.reduce_sum(portfolio_values, axis=2) + cash
        net_portfolio_values = net_portfolio_values.numpy()[0]

        # Plot the portfolio value over time
        timesteps = np.arange(self.N + 1)
        fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
        agent_color = self._get_agent_plot_color(self.agent, 0)
        ax1.bar(timesteps, net_portfolio_values, label="Valor Neto del Portafolio", color=agent_color, alpha=0.6)

        # Plot the contingent claim payoff at the final timestep as a bar
        ax1.bar(self.N, payoff.numpy()[0], color='r', alpha=0.6, label="Payoff del Claim Contingente")

        ax1.set_xlabel("Paso de Tiempo")
        ax1.set_ylabel("Valor del Portafolio", color=agent_color)
        ax1.tick_params(axis='y', labelcolor=agent_color)

        # Create a secondary y-axis for the underlying asset price
        ax2 = ax1.twinx()
        ax2.plot(timesteps, single_path.numpy()[0, :, 0], label="Trayectoria del Subyacente", color='g', alpha=0.6)
        ax2.set_ylabel("Precio del Subyacente", color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.axhline(y=self.contingent_claim.strike, color='orange', linestyle='-', label="Precio de Strike")

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # Title and layout
        plt.title(f"Estrategia de Cobertura en el Tiempo. PnL: {pnl.numpy()[0]:.4f}")
        fig.tight_layout()

        # Save the plot if a path is provided, otherwise show it
        if save_plot_path:
            plt.savefig(save_plot_path)
            print(f"Plot saved to {save_plot_path}")
        else:
            plt.show()

    def compare_hedging_strategy(self, agents, n_paths=1, random_seed=None, 
                                save_plot_path=None, language='es'):
        """
        Compare the hedging strategies of multiple agents by plotting their actions (deltas) over time alongside the stock price.

        Arguments:
        - agents (list): List of agent instances to compare.
        - n_paths (int): Number of paths to generate for plotting. Default is 1.
        - random_seed (int, optional): Seed for random number generator to ensure reproducibility.
        - save_plot_path (str, optional): File path to save the plot. If None, the plot is only shown.
        - language (str): Language code for agent labels and plot texts (e.g., 'en' for English, 'es' for Spanish).

        Returns:
        - None
        """

        if n_paths < 1:
            raise ValueError("n_paths must be at least 1.")

        # Define multilingual titles and labels
        plot_titles = {
            'en': "Comparison of Hedging Strategies",
            'es': "Comparación de Estrategias de Cobertura"
        }

        axis_labels = {
            'xlabel': {
                'en': "Timestep",
                'es': "Paso de Tiempo"
            },
            'ylabel_deltas': {
                'en': "Hedging Actions (Delta)",
                'es': "Acciones de Cobertura (Delta)"
            },
            'ylabel_stock': {
                'en': "Underlying Asset Price",
                'es': "Precio del Activo Subyacente"
            },
            'stock_label': {
                'en': "Stock Price",
                'es': "Precio de la Acción"
            }
        }

        legend_labels = {
            'actions': {
                'en': "Actions",
                'es': "Acciones"
            },
            'stock': {
                'en': "Stock Price",
                'es': "Precio de la Acción"
            }
        }

        # Generate the specified number of paths
        paths = self.generate_data(n_paths, random_seed=random_seed)  # Shape: (n_paths, N+1, n_instruments)
        
        # For simplicity, we'll plot the first path
        path = paths[0]  # Shape: (N+1, n_instruments)
        stock_prices = path[:, 0].numpy()  # Assuming the first instrument is the stock

        timesteps = np.arange(self.N + 1)

        plt.figure(figsize=(12, 8))

        # Initialize primary axis for deltas
        ax1 = plt.gca()
        ax1.set_xlabel(axis_labels['xlabel'].get(language, 'Timestep'), fontsize=14)
        ax1.set_ylabel(axis_labels['ylabel_deltas'].get(language, 'Hedging Actions (Delta)'), fontsize=14)
        ax1.set_title(plot_titles.get(language, "Comparison of Hedging Strategies"), fontsize=16)
        
        # Iterate over each agent and plot their actions (deltas)
        for agent in agents:
            # Get T_minus_t for the path
            T_minus_t = self.get_T_minus_t(1)  # Shape: (1, N)
            
            # Process the batch to get actions
            actions = agent.process_batch(path[tf.newaxis, ...], T_minus_t)  # Shape: (1, N, n_instruments)
            actions = actions.numpy()[0, :self.N, 0]  # Assuming actions on the first instrument

            # Prepend a zero action for the initial time step
            actions = np.insert(actions, 0, 0)

            # Plot the actions (deltas) as step plots
            ax1.step(
                timesteps,
                actions,
                where='post',
                label=f"{self._get_agent_plot_name(agent, language)} {legend_labels['actions'].get(language, 'Actions')}",
                alpha=0.7,
                color=self._get_agent_plot_color(agent)
            )

        # Initialize secondary axis for stock price
        ax2 = ax1.twinx()
        ax2.set_ylabel(axis_labels['ylabel_stock'].get(language, 'Underlying Asset Price'), fontsize=14)
        
        # Plot the stock price on the secondary y-axis
        ax2.plot(timesteps, stock_prices, label=legend_labels['stock'].get(language, 'Stock Price'), color='grey', linestyle='--', linewidth=2)

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.grid(True, linestyle='--', alpha=0.7)

        if save_plot_path:
            directory = os.path.dirname(save_plot_path)
            filename = os.path.basename(save_plot_path)
            
            if os.path.isdir(save_plot_path):
                raise IsADirectoryError(f"The path '{save_plot_path}' is a directory. Please provide a valid file path.")

            print(f"Attempting to save plot to: {save_plot_path}")
            os.makedirs(directory, exist_ok=True)
            plt.savefig(save_plot_path, bbox_inches='tight')
            print(f"Comparison plot saved to {save_plot_path}")
        else:
            plt.show()

    def bootstrap_confidence_intervals(
        self,
        agents,
        statistics, 
        n_paths=10_000, 
        n_bootstraps=1_000,
        confidence_level=0.95,
        random_seed=None,
        plot_histograms=False,
        save_plot_dir=None,
        language='es',
        save_actions_path=None,
        fixed_actions_paths=None,
        pricing_method='fixed',
        batch_size=100,  # New parameter for batch processing
        progress_log_every_batches=None,
        progress_log_every_seconds=15.0,
    ):
        """
        Compute bootstrap confidence intervals for given statistics applied to the PnL distribution of multiple agents.

        Parameters:
        - agents (list): List of agent instances to evaluate.
        - statistics (callable or list of callables): One or multiple statistics to apply to the PnL distribution.
        Each callable should accept a 1D array of pnl values and return a scalar.
        - n_paths (int): Number of paths used to generate PnL samples (default: 10,000).
        - n_bootstraps (int): Number of bootstrap samples (default: 1,000).
        - confidence_level (float): Confidence level for the interval (default: 0.95).
        - random_seed (int): Random seed for reproducibility (default: None).
        - plot_histograms (bool): If True, plot bootstrap distributions for each statistic and agent (default: False).
        - save_plot_dir (str): Directory to save the histograms if plot_histograms is True (default: None).
        - language (str): Language for labels (default: 'en').
        - save_actions_path (str): If provided, directory to save actions of each agent (default: None).
        - fixed_actions_paths (dict): If provided, dictionary mapping agent_name to path of precomputed actions.
        - pricing_method (str): 'fixed' or 'individual', same as terminal_hedging_error_multiple_agents method.
        - batch_size (int): Number of bootstrap samples to process in each batch (default: 100).
        - progress_log_every_batches (int|None): If provided, emits bootstrap progress every N batches.
          If None, uses an automatic interval based on total batches.
        - progress_log_every_seconds (float|None): Emit progress at least every N seconds during bootstrap.
          If None or <= 0, disables time-based periodic logs.

        Returns:
        - results_df (pd.DataFrame): A DataFrame with agents as rows and statistics & confidence intervals as columns.
        """
        # Ensure statistics is a list
        if not isinstance(statistics, list):
            statistics = [statistics]

        # Generate the data paths
        paths = self.generate_data(n_paths, random_seed=random_seed)

        # Compute prices based on the pricing_method
        if pricing_method == 'fixed':
            first_agent = agents[0]
            price = self.get_agent_price(first_agent, n_paths=n_paths, random_seed=random_seed)
            prices = [price] * len(agents)
            print(f"Using fixed price from the first agent: {price}")
        elif pricing_method == 'individual':
            prices = []
            for agent in agents:
                price = self.get_agent_price(agent, n_paths=n_paths, random_seed=random_seed)
                prices.append(price)
                print(f"Computed price for agent '{agent.name}': {price}")
        else:
            raise ValueError(f"Invalid pricing_method '{pricing_method}'. Choose 'fixed' or 'individual'.")

        # Ensure save_actions_path exists if provided
        if save_actions_path:
            os.makedirs(save_actions_path, exist_ok=True)

        # If fixed_actions_paths is not provided, check for existing action files in save_actions_path
        if fixed_actions_paths is None and save_actions_path:
            fixed_actions_paths = {}
            for idx, agent in enumerate(agents):
                agent_id = self._get_agent_identifier(agent, idx)
                agent_actions_filename = f"{agent_id}_actions.npy"
                agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                if os.path.isfile(agent_actions_path):
                    fixed_actions_paths[agent_id] = agent_actions_path
                    print(f"Using existing actions file for agent '{agent_id}' at '{agent_actions_path}'.")
            # If no existing files are found, set fixed_actions_paths back to None
            if not fixed_actions_paths:
                fixed_actions_paths = None

        # Prepare results storage
        columns = ['Agent']
        for stat_fn in statistics:
            stat_name = stat_fn.name
            columns.extend([
                f"{stat_name}_point_estimate", 
                f"{stat_name}_ci_lower", 
                f"{stat_name}_ci_upper"
            ])

        results = []

        # Seed for bootstrap reproducibility
        rng = np.random.default_rng(random_seed)

        # Create plot directory if needed
        if plot_histograms and save_plot_dir:
            os.makedirs(save_plot_dir, exist_ok=True)

        # Precompute bootstrap indices in batches
        n_batches = int(np.ceil(n_bootstraps / batch_size))
        if progress_log_every_batches is None:
            progress_log_every_batches = max(1, n_batches // 20)  # ~5% checkpoints
        progress_log_every_batches = max(1, int(progress_log_every_batches))

        # Process each agent with a progress bar
        for idx, agent in enumerate(tqdm(agents, desc="Processing agents")):
            agent_start_time = time.perf_counter()
            agent_name = agent.name
            agent_id = self._get_agent_identifier(agent, idx)
            price = prices[idx]
            tqdm.write(
                f"[bootstrap] agent={agent_id}: starting "
                f"(paths={n_paths}, bootstraps={n_bootstraps}, batches={n_batches}, batch_size={batch_size})"
            )

            # Check if actions are fixed for this agent
            fixed_path = None
            if fixed_actions_paths:
                if agent_id in fixed_actions_paths:
                    fixed_path = fixed_actions_paths[agent_id]
                elif agent_name in fixed_actions_paths:  # backward compatibility
                    fixed_path = fixed_actions_paths[agent_name]
            if fixed_path is not None:
                if not os.path.isfile(fixed_path):
                    raise FileNotFoundError(f"Fixed actions file for agent '{agent_id}' not found at '{fixed_path}'.")
                # Load val_actions from .npy
                print(f"Loading fixed actions for agent '{agent_id}' from '{fixed_path}'.")
                val_actions_np = np.load(fixed_path)
                val_actions = tf.convert_to_tensor(val_actions_np, dtype=tf.float32)
            else:
                # Process batch to get val_actions
                T_minus_t = self.get_T_minus_t(paths.shape[0])
                val_actions = agent.process_batch(paths, T_minus_t)

                # Save val_actions if save_actions_path is provided
                if save_actions_path:
                    agent_actions_filename = f"{agent_id}_actions.npy"
                    agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                    # Convert TensorFlow tensor to NumPy array
                    val_actions_np = val_actions.numpy()
                    # Save as .npy
                    np.save(agent_actions_path, val_actions_np)
                    print(f"Saved val_actions for agent '{agent_id}' to '{agent_actions_path}'.")

            # Calculate PnL and error
            pnl = self.calculate_pnl(paths, val_actions)
            # Adjusting for present value
            error = price + pnl * np.exp(-self.r * self.T)
            error_np = error.numpy().astype(np.float32)  # Use float32 to save memory
            tqdm.write(
                f"[bootstrap] agent={agent_id}: pnl/error ready "
                f"(n_errors={error_np.shape[0]})"
            )

            # Initialize containers for statistics
            point_estimates = {}
            bootstrap_stats = {stat_fn: [] for stat_fn in statistics}

            # Compute point estimates
            for stat_fn in statistics:
                stat_name = stat_fn.name
                point_estimate = stat_fn(error_np)
                if isinstance(point_estimate, tf.Tensor):
                    point_estimate = point_estimate.numpy().item()
                point_estimates[stat_name] = point_estimate

            # Perform bootstrapping in batches to reduce memory usage
            last_progress_log_time = time.perf_counter()
            for batch_idx in range(n_batches):
                current_batch_size = min(batch_size, n_bootstraps - batch_idx * batch_size)
                # Generate bootstrap indices for the current batch
                bootstrap_indices = rng.integers(0, n_paths, size=(current_batch_size, n_paths))
                # Extract bootstrap samples
                bootstrap_samples = error_np[bootstrap_indices]  # Shape: (current_batch_size, n_paths)

                # Compute statistics for the current batch
                for stat_fn in statistics:
                    stat_name = stat_fn.name
                    # Try vectorized call first. If it returns a scalar/wrong shape, fallback per sample.
                    stat_values = None
                    try:
                        candidate = stat_fn(bootstrap_samples)
                        if isinstance(candidate, tf.Tensor):
                            candidate = candidate.numpy()
                        candidate = np.asarray(candidate)
                        if candidate.ndim == 0 or candidate.shape[0] != current_batch_size:
                            raise ValueError(
                                f"Non-vectorized output shape for {stat_name}: {candidate.shape}"
                            )
                        stat_values = candidate.reshape(-1)
                    except Exception:
                        # Fallback: evaluate one bootstrap sample at a time.
                        stat_values = np.array(
                            [
                                stat_fn(sample).numpy()
                                if isinstance(stat_fn(sample), tf.Tensor)
                                else stat_fn(sample)
                                for sample in bootstrap_samples
                            ],
                            dtype=np.float32,
                        ).reshape(-1)

                    bootstrap_stats[stat_fn].extend(stat_values.tolist())

                batches_done = batch_idx + 1
                should_log_batch = (batches_done % progress_log_every_batches == 0) or (batches_done == n_batches)
                should_log_time = False
                if progress_log_every_seconds is not None and progress_log_every_seconds > 0:
                    now = time.perf_counter()
                    should_log_time = (now - last_progress_log_time) >= float(progress_log_every_seconds)
                else:
                    now = time.perf_counter()
                if should_log_batch or should_log_time:
                    elapsed = now - agent_start_time
                    processed_bootstraps = min(batches_done * batch_size, n_bootstraps)
                    progress_ratio = processed_bootstraps / float(n_bootstraps)
                    eta_seconds = np.nan
                    if progress_ratio > 0:
                        eta_seconds = elapsed * (1.0 - progress_ratio) / progress_ratio
                    eta_display = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "n/a"
                    tqdm.write(
                        f"[bootstrap] agent={agent_id}: "
                        f"{processed_bootstraps}/{n_bootstraps} samples "
                        f"({progress_ratio*100:.1f}%), elapsed={elapsed:.1f}s, eta={eta_display}"
                    )
                    last_progress_log_time = now

            # Compute confidence intervals
            agent_result = {'Agent': self._get_agent_plot_name(agent, language)}
            for stat_fn in statistics:
                stat_name = stat_fn.name
                stats = np.array(bootstrap_stats[stat_fn], dtype=np.float32)
                lower_bound = np.percentile(stats, 100 * (1 - confidence_level) / 2)
                upper_bound = np.percentile(stats, 100 * (1 - (1 - confidence_level) / 2))
                point_estimate = point_estimates[stat_name]

                agent_result[f"{stat_name}_point_estimate"] = point_estimate
                agent_result[f"{stat_name}_ci_lower"] = lower_bound
                agent_result[f"{stat_name}_ci_upper"] = upper_bound

                if plot_histograms:
                    stat_plot_name = getattr(stat_fn, 'plot_name', stat_fn.__name__)
                    agent_plot_name = self._get_agent_plot_name(agent, language)
                    plot_labels = {
                        'en': {
                            'title': f"Bootstrap Distribution of {stat_plot_name} for {agent_plot_name}",
                            'xlabel': f"{stat_plot_name}",
                            'ylabel': "Frequency",
                            'point_estimate': "Point Estimate",
                            'ci_lower': f"{int(confidence_level*100)}% CI Lower",
                            'ci_upper': f"{int(confidence_level*100)}% CI Upper"
                        },
                        'es': {
                            'title': f"Distribución Bootstrap de '{stat_plot_name}' para errores del {agent_plot_name}",
                            'xlabel': f"{stat_plot_name}",
                            'ylabel': "Frecuencia",
                            'point_estimate': "Estimación puntual",
                            'ci_lower': f"Límite inferior IC {int(confidence_level*100)}%",
                            'ci_upper': f"Límite superior IC {int(confidence_level*100)}%"
                        }
                    }

                    labels = plot_labels.get(language, plot_labels['en'])

                    plt.figure(figsize=(10, 6))
                    plt.hist(bootstrap_stats[stat_fn], bins=30, alpha=0.7, edgecolor='black')
                    plt.title(labels['title'])
                    plt.xlabel(labels['xlabel'])
                    plt.ylabel(labels['ylabel'])

                    plt.axvline(point_estimate, color='red', linestyle='--', label=labels['point_estimate'])
                    plt.axvline(lower_bound, color='green', linestyle='--', label=labels['ci_lower'])
                    plt.axvline(upper_bound, color='green', linestyle='--', label=labels['ci_upper'])

                    plt.legend()
                    if save_plot_dir:
                        histogram_path = os.path.join(
                            save_plot_dir,
                            f"{agent_id}_{stat_plot_name}_bootstrap_histogram.pdf"
                        )
                        plt.savefig(histogram_path)
                        print(f"Histogram saved to {histogram_path}")
                    else:
                        plt.show()
                    plt.close()

            # Append the agent's results to the results list
            results.append(agent_result)
            agent_elapsed = time.perf_counter() - agent_start_time
            tqdm.write(f"[bootstrap] agent={agent_id}: completed in {agent_elapsed:.1f}s")

        # Create DataFrame from results
        results_df = pd.DataFrame(results, columns=columns)
        return results_df
