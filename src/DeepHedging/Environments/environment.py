import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings
import pandas as pd

class Environment:
    def __init__(self, agent, T, N, r, instrument_list, n_instruments, contingent_claim, cost_function, risk_measure,
                 n_epochs, batch_size, learning_rate, optimizer):
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
        self.optimizer = optimizer(learning_rate=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def generate_data(self, n_paths, random_seed=None):

        data = tf.TensorArray(dtype=tf.float32, size=self.n_instruments)
        
        i = 0
        for instrument in self.instrument_list:
            instrument_data = instrument.generate_paths(n_paths, random_seed=random_seed)
            if isinstance(instrument_data, tuple):
                for individual_data in instrument_data:
                    data = data.write(i, individual_data)
                    i += 1
            else:
                data = data.write(i, instrument_data)
                i += 1
        
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

        # Calculate the payoff of the contingent claim
        payoff = self.contingent_claim.calculate_payoff(paths[:, :, 0]) # ASSUMPTION: The CC is calculated based on the first instrument
        
        # Calculate the PnL
        pnl = final_portfolio_value + final_cash_positions - payoff
        
        if include_decomposition:
            return pnl, portfolio_values, cash, payoff
        else:
            return pnl

    def loss_function(self, paths, actions):
        pnl = self.calculate_pnl(paths, actions)
        return self.risk_measure.calculate(pnl)

    def train(self, train_paths, val_paths = 0):

        train_data = self.generate_data(train_paths) # (n_paths, N+1, n_instruments)
        T_minus_t = self.get_T_minus_t(self.batch_size) # (n_paths, N)
    
        if val_paths > 0:
            val_data = self.generate_data(val_paths)
            T_minus_t_val =  self.get_T_minus_t(val_paths)

        for epoch in range(self.n_epochs):
            # Training
            epoch_losses = []
            for i in range(0, train_paths, self.batch_size):
                batch_paths = train_data[i:i+self.batch_size]
                loss = self.agent.train_batch(batch_paths, T_minus_t, self.optimizer, self.loss_function)
                epoch_losses.append(loss.numpy())

            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)

            # Validation
            if val_paths > 0:
                val_actions = self.agent.process_batch(val_data, T_minus_t_val)
                val_loss = self.loss_function(val_data, val_actions)
                self.val_losses.append(val_loss.numpy())
                print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.learning_rate}")
            else:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.4f}, LR: {self.optimizer.learning_rate}")

        if val_paths > 0:
            return self.train_losses, self.val_losses
        else:
            return self.train_losses

    def test(self, paths_to_test = None, n_paths = None, random_seed = None, 
             plot_pnl = False, plot_title = 'PnL distribution', save_plot_path = None):
        
        if paths_to_test:
            paths = paths_to_test
        elif n_paths:
            paths = self.generate_paths(n_paths, random_seed = random_seed)
        else:
            raise ValueError('Insert either paths_to_test or n_paths.')
        
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
            plt.title(f'{plot_title}. Loss: {loss:.4f}', fontsize=14)
            plt.xlabel('PnL', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)

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

    def terminal_hedging_error_multiple_agents(self, agents, n_paths=10_000, random_seed=None, 
            plot_error=False, plot_title='Terminal Hedging Error', save_plot_path=None, 
            colors=None, save_stats_path=None, loss_functions=None, min_x=-0.3, max_x=0.3, language='en',
            save_actions_path=None, fixed_actions_paths=None, pricing_method='fixed'):
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
        agents = np.unique(agents)
        # Generate the data paths
        paths = self.generate_data(n_paths, random_seed=random_seed)

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
            for agent in agents:
                agent_actions_filename = f"{agent.name}_actions.npy"
                agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                if os.path.isfile(agent_actions_path):
                    fixed_actions_paths[agent.name] = agent_actions_path
                    print(f"Using existing actions file for agent '{agent.name}' at '{agent_actions_path}'.")
            # If no existing files are found, set fixed_actions_paths back to None
            if not fixed_actions_paths:
                fixed_actions_paths = None

        for idx, agent in enumerate(agents):
            agent_name = agent.name
            price = prices[idx]

            # Check if actions are fixed for this agent
            if fixed_actions_paths and agent_name in fixed_actions_paths:
                fixed_path = fixed_actions_paths[agent_name]
                if not os.path.isfile(fixed_path):
                    raise FileNotFoundError(f"Fixed actions file for agent '{agent_name}' not found at '{fixed_path}'.")
                # Load val_actions from .npy
                print(f"Loading fixed actions for agent '{agent_name}' from '{fixed_path}'.")
                val_actions_np = np.load(fixed_path)
                val_actions = tf.convert_to_tensor(val_actions_np, dtype=tf.float32)
            else:
                # Process batch to get val_actions
                T_minus_t = self.get_T_minus_t(paths.shape[0])
                val_actions = agent.process_batch(paths, T_minus_t)

                # Save val_actions if save_actions_path is provided
                if save_actions_path:
                    agent_actions_filename = f"{agent_name}_actions.npy"
                    agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                    # Convert TensorFlow tensor to NumPy array
                    val_actions_np = val_actions.numpy()
                    # Save as .npy
                    np.save(agent_actions_path, val_actions_np)
                    print(f"Saved val_actions for agent '{agent_name}' to '{agent_actions_path}'.")

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

        if plot_error:
            plt.figure(figsize=(10, 6))

            # Define the bins for the histogram
            bins = np.linspace(min_x, max_x, 60)  # 60 bins across the specified range

            # Set default colors if not provided
            if colors is None:
                cmap = plt.cm.get_cmap('tab10', len(agents))
                colors = [cmap(i) for i in range(len(agents))]

            # Plot each agent's error histogram
            for i, error in enumerate(errors):
                plt.hist(error, bins=bins, density=True, color=colors[i], alpha=0.6, edgecolor='black', 
                        label=f'{agents[i].plot_name.get(language)}')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(plot_title, fontsize=14)
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

            plt.show()

        # Save statistics to Excel if save_stats_path is provided
        if save_stats_path:
            os.makedirs(os.path.dirname(save_stats_path), exist_ok=True)
            data = {
                'Agent': [agent.plot_name.get(language) for agent in agents],
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
            return df

        return mean_errors, std_errors, loss_results if loss_functions else None

    def aterminal_hedging_error_multiple_agents(self, agents, n_paths=10_000, random_seed=None, 
        fixed_price=None, plot_error=False, plot_title='Terminal Hedging Error', save_plot_path=None, 
        colors=None, save_stats_path=None, loss_functions=None, min_x=-0.3, max_x=0.3, language='en',
        save_actions_path=None, fixed_actions_paths=None):
        """
        Computes terminal hedging error for multiple agents, generates plots, and saves statistics.

        Arguments:
        - agents (list): List of agent instances to evaluate.
        - n_paths (int): Number of paths to generate for evaluation.
        - random_seed (int or None): Random seed for path generation.
        - fixed_price (float): The fixed price to use in calculating the hedging error.
        - plot_error (bool): Whether to plot the error histograms.
        - plot_title (str): Title of the plot.
        - save_plot_path (str or None): Path to save the plot image. If None, plot is shown.
        - colors (list or None): List of colors for the plot. If None, default colors are used.
        - save_stats_path (str or None): Path to save the statistics as an Excel file.
        - loss_functions (list of callables or None): List of loss functions to evaluate on the PnL.
        - min_x (float): Minimum x-axis value for the histogram.
        - max_x (float): Maximum x-axis value for the histogram.
        - language (str): Language for labels ('en' for English, 'es' for Spanish).
        - save_actions_path (str or None): Directory path to save each agent's val_actions as `.npy` files.
        - fixed_actions_paths (dict or None): Dictionary mapping agent names to fixed actions file paths.
                                            Example: {'agent1': 'path/to/agent1_actions.npy', ...}

        Returns:
        - mean_errors (list): List of mean errors for each agent.
        - std_errors (list): List of standard deviations of errors for each agent.
        - losses (dict or None): Dictionary of loss function results for each agent.
        """

        # Generate the data paths
        paths = self.generate_data(n_paths, random_seed=random_seed)

        # Validate fixed_price
        if fixed_price is not None:
            price = fixed_price
        else:
            raise ValueError('Insert fixed_price.')

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
            for agent in agents:
                agent_actions_filename = f"{agent.name}_actions.npy"
                agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                if os.path.isfile(agent_actions_path):
                    fixed_actions_paths[agent.name] = agent_actions_path
                    print(f"Using existing actions file for agent '{agent.name}' at '{agent_actions_path}'.")
            # If no existing files are found, set fixed_actions_paths back to None
            if not fixed_actions_paths:
                fixed_actions_paths = None
        evaluated_agents = []
        for agent in agents:
            agent_name = agent.name
            if agent_name in evaluated_agents:
                continue

            # Check if actions are fixed for this agent
            if fixed_actions_paths and agent_name in fixed_actions_paths:
                fixed_path = fixed_actions_paths[agent_name]
                if not os.path.isfile(fixed_path):
                    raise FileNotFoundError(f"Fixed actions file for agent '{agent_name}' not found at '{fixed_path}'.")
                # Load val_actions from .npy
                print(f"Loading fixed actions for agent '{agent_name}' from '{fixed_path}'.")
                val_actions_np = np.load(fixed_path)
                val_actions = tf.convert_to_tensor(val_actions_np, dtype=tf.float32)
            else:
                # Process batch to get val_actions
                T_minus_t = self.get_T_minus_t(paths.shape[0])
                val_actions = agent.process_batch(paths, T_minus_t)

                # Save val_actions if save_actions_path is provided
                if save_actions_path:
                    agent_actions_filename = f"{agent_name}_actions.npy"
                    agent_actions_path = os.path.join(save_actions_path, agent_actions_filename)
                    # Convert TensorFlow tensor to NumPy array
                    val_actions_np = val_actions.numpy()
                    # Save as .npy
                    np.save(agent_actions_path, val_actions_np)
                    print(f"Saved val_actions for agent '{agent_name}' to '{agent_actions_path}'.")

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

        if plot_error:
            plt.figure(figsize=(10, 6))

            # Define the bins for the histogram
            bins = np.linspace(min_x, max_x, 60)  # 60 bins across the specified range

            # Set default colors if not provided
            if colors is None:
                cmap = plt.cm.get_cmap('tab10', len(agents))
                colors = [cmap(i) for i in range(len(agents))]

            # Plot each agent's error histogram
            for i, error in enumerate(errors):
                plt.hist(error, bins=bins, density=True, color=colors[i], alpha=0.6, edgecolor='black', 
                        label=f'{agents[i].plot_name.get(language)}')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(plot_title, fontsize=14)
            plt.xlabel('Error', fontsize=12)

            # Set y-axis label based on language
            if language == 'es':
                plt.ylabel('Densidad', fontsize=12)
            else:
                plt.ylabel('Density', fontsize=12)

            plt.legend()

            # Save or show the plot
            if save_plot_path:
                os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
                plt.savefig(save_plot_path)
                print(f"Plot saved to {save_plot_path}")

            plt.show()

        # Save statistics to Excel if save_stats_path is provided
        if save_stats_path:
            os.makedirs(os.path.dirname(save_stats_path), exist_ok=True)
            data = {
                'Agent': [agent.plot_name.get(language) for agent in agents],
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
            return df

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
        single_path = self.generate_paths(1, random_seed = random_seed)  # Shape: (1, N+1)
        T_minus_t_single = tf.range(self.N, 0, -1, dtype=tf.float32) * self.dt
        T_minus_t_single = tf.expand_dims(T_minus_t_single, axis=0)  # Shape: (1, N)

        # Get the hedging actions
        actions = self.agent.process_batch(single_path, T_minus_t_single)  # Shape: (1, N)

        pnl, portfolio_values, cash, payoff = self.calculate_pnl(single_path, actions, include_decomposition = True)

        net_portfolio_values = portfolio_values + cash
        net_portfolio_values = net_portfolio_values.numpy()[0]

        # Plot the portfolio value over time
        timesteps = np.arange(self.N + 1)
        fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
        ax1.bar(timesteps, net_portfolio_values, label="Net Portfolio Values", color='b', alpha=0.6)

        # Plot the contingent claim payoff at the final timestep as a bar
        ax1.bar(self.N, payoff, color='r', alpha=0.6, label="Contingent Claim Payoff")

        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Portfolio Value", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a secondary y-axis for the underlying asset price
        ax2 = ax1.twinx()
        ax2.plot(timesteps, single_path.numpy().flatten(), label="Underlying Asset Path", color='g', alpha=0.6)
        ax2.set_ylabel("Underlying Asset Price", color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.axhline(y=self.contingent_claim.strike, color='orange', linestyle='-', label="Strike Price")

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # Title and layout
        plt.title(f"Hedging Strategy Over Time. PnL: {pnl.numpy()[0]:.4f}")
        fig.tight_layout()

        # Save the plot if a path is provided, otherwise show it
        if save_plot_path:
            plt.savefig(save_plot_path)
            print(f"Plot saved to {save_plot_path}")
        else:
            plt.show()

