import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings
import pandas as pd

# Import DeepHedging modules (Ensure these are available in your environment)
from DeepHedging.Agents import (BaseAgent, SimpleAgent, RecurrentAgent, LSTMAgent, 
                                GRUAgent, WaveNetAgent, DeltaHedgingAgent, 
                                GeometricAsianDeltaHedgingAgent, GeometricAsianDeltaHedgingAgent2, 
                                GeometricAsianNumericalDeltaHedgingAgent, QuantlibAsianGeometricAgent, 
                                ArithmeticAsianMonteCarloAgent, ArithmeticAsianControlVariateAgent, 
                                MonteCarloAgent)

from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.ContingentClaims import (
    EuropeanCall, EuropeanPut, AsianGeometricCall, AsianGeometricPut,
    AsianArithmeticCall, AsianArithmeticPut
)
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase, Mean, StdDev
from DeepHedging.Environments import Environment

AGENTS = {
    'SimpleAgent': SimpleAgent,
    'RecurrentAgent': RecurrentAgent,
    'LSTMAgent': LSTMAgent,
    'GRUAgent': GRUAgent,
    'WaveNetAgent': WaveNetAgent,
    'DeltaHedgingAgent': DeltaHedgingAgent,
    'GeometricAsianDeltaHedgingAgent': GeometricAsianDeltaHedgingAgent,
    'GeometricAsianDeltaHedgingAgent2': GeometricAsianDeltaHedgingAgent2,
    'GeometricAsianNumericalDeltaHedgingAgent': GeometricAsianNumericalDeltaHedgingAgent,
    'QuantlibAsianGeometricAgent': QuantlibAsianGeometricAgent,
    'ArithmeticAsianMonteCarloAgent': ArithmeticAsianMonteCarloAgent,
    'ArithmeticAsianControlVariateAgent': ArithmeticAsianControlVariateAgent,
    'MonteCarloAgent': MonteCarloAgent
}

def parse_agent_model_names(arg_list):
    """
    Parses a list of strings in the format 'agent_name=model_name' into a dictionary.
    """
    agent_model_names = {}
    if arg_list:
        for item in arg_list:
            try:
                agent_name, model_name = item.split('=')
                agent_model_names[agent_name] = model_name
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid format for agent_model_names: '{item}'. "
                    "Expected format 'agent_name=model_name'"
                )
    if agent_model_names == {}:
        agent_model_names = None
    return agent_model_names

def parse_fixed_actions_paths(arg_list):
    """
    Parses a list of strings in the format 'agent_name=path/to/actions.csv' into a dictionary.
    """
    fixed_paths = {}
    if arg_list:
        for item in arg_list:
            try:
                agent_name, path = item.split('=')
                fixed_paths[agent_name] = path
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid format for fixed_actions_paths: '{item}'. "
                    "Expected format 'agent_name=path/to/actions.csv'"
                )
    if fixed_paths == {}:
        fixed_paths = None
    return fixed_paths

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Deep Hedging agents with bootstrap confidence intervals.")

    # Simulation parameters
    parser.add_argument('--T', type=float, default=22/365, help='Time to maturity (default: 22/365)')
    parser.add_argument('--N', type=int, default=22, help='Number of time steps (default: 22)')
    parser.add_argument('--r', type=float, default=0.05, help='Risk-free rate (default: 0.05)')
    parser.add_argument('--S0', type=float, default=100, help='Initial stock price (default: 100)')
    parser.add_argument('--sigma', type=float, default=0.05, help='Volatility (default: 0.05)')
    parser.add_argument('--strike', type=float, default=100, help='Strike price (default: 100)')

    # Contingent claim parameters
    parser.add_argument('--contingent_claim', type=str, default='AsianGeometricCall',
                        choices=[
                            'EuropeanCall', 'EuropeanPut', 'AsianGeometricCall', 
                            'AsianGeometricPut', 'AsianArithmeticCall', 'AsianArithmeticPut'
                        ],
                        help='Type of contingent claim (default: AsianGeometricCall)')

    # Agent parameters
    parser.add_argument('--agents', type=str, nargs='+', required=True,
                        choices=list(AGENTS.keys()),
                        help='List of agents to evaluate')
    parser.add_argument('--bump_size', type=float, default=0.001, help='Bump size for numerical delta (default: 0.001)')

    # Environment parameters
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size (default: 2000)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')

    # Evaluation parameters
    parser.add_argument('--n_paths', type=int, default=100000, help='Number of paths for evaluation (default: 100000)')
    parser.add_argument('--random_seed', type=int, default=33, help='Random seed for reproducibility (default: 33)')

    # Cost function parameters
    parser.add_argument('--proportional_cost', type=float, default=0.0, help='Proportional cost (default: 0.0)')

    # Risk measure parameters
    parser.add_argument('--cvar_alpha', type=float, default=0.5, help='CVaR alpha (default: 0.5)')

    # Plot parameters
    parser.add_argument('--language', type=str, choices=['en', 'es'], default='es', help='Language for labels: "en" or "es" (default: es)')

    # File paths
    parser.add_argument('--model_name', type=str, default='asian_1', help='Model name (default: asian_1)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to load models from (default: models)')
    parser.add_argument('--optimizers_dir', type=str, default='optimizers', help='Directory to load optimizers from (default: optimizers)')
    parser.add_argument('--save_plots_dir', type=str, default='assets/plots', help='Directory to save plots (default: assets/plots)')
    parser.add_argument('--save_stats_dir', type=str, default='assets/csvs', help='Directory to save statistics (default: assets/csvs)')
    parser.add_argument('--save_actions_path', type=str, default=None, 
                        help='Directory to save each agent\'s actions as npy files (default: None)')
    parser.add_argument('--fixed_actions_paths', type=str, nargs='*', default=None,
                        help='Fixed actions paths in the format agent_name=path/to/actions.npy. '
                             'Example: agent1=path/to/agent1_actions.npy agent2=path/to/agent2_actions.npy')

    # New argument for agent-specific model names
    parser.add_argument('--agent_model_names', type=str, nargs='*', default=None,
                        help='Agent model names in the format agent_name=model_name. '
                             'Example: agent1=model1 agent2=model2')

    # New argument for pricing method
    parser.add_argument('--pricing_method', type=str, choices=['fixed', 'individual'], default='fixed',
                        help='Pricing method: "fixed" to charge all agents the price of the first agent, '
                             '"individual" to charge each agent their own price (default: fixed)')

    # Bootstrap parameters
    parser.add_argument('--n_bootstraps', type=int, default=1000, help='Number of bootstrap samples (default: 1000)')
    parser.add_argument('--confidence_level', type=float, default=0.95, help='Confidence level for intervals (default: 0.95)')
    parser.add_argument('--plot_histograms', action='store_true', help='If set, plot bootstrap histograms.')
    parser.add_argument('--bootstrap_plots_dir', type=str, default='assets/bootstrap_plots', help='Directory to save bootstrap histograms (default: assets/bootstrap_plots)')

    # Statistics to compute
    parser.add_argument('--statistics', type=str, nargs='+', default=['mean','std'],
                        help='Statistics to compute. Possible values: mean, std, median. (default: mean std)')

    return parser.parse_args()

def get_agent(agent_name, instrument, contingent_claim, path_transformation_configs=None, n_hedging_timesteps=None, **kwargs):

    if agent_name not in AGENTS:
        raise ValueError(f"Agent '{agent_name}' is not recognized. Available agents: {list(AGENTS.keys())}")
    agent_class = AGENTS[agent_name]
    if agent_class.is_trainable:
        path_transformation_configs = [{'transformation_type': 'log_moneyness', 'K': contingent_claim.strike}]
        return agent_class(path_transformation_configs=path_transformation_configs,
                            n_hedging_timesteps=n_hedging_timesteps)
    return agent_class(instrument, contingent_claim, **kwargs)

def get_contingent_claim(claim_type, strike):
    claims = {
        'EuropeanCall': EuropeanCall(strike=strike),
        'EuropeanPut': EuropeanPut(strike=strike),
        'AsianGeometricCall': AsianGeometricCall(strike=strike),
        'AsianGeometricPut': AsianGeometricPut(strike=strike),
        'AsianArithmeticCall': AsianArithmeticCall(strike=strike),
        'AsianArithmeticPut': AsianArithmeticPut(strike=strike)
    }
    if claim_type not in claims:
        raise ValueError(f"Contingent Claim '{claim_type}' is not recognized. Available claims: {list(claims.keys())}")
    return claims[claim_type]

def get_statistics(statistics):
    stats_map = {
        'mean': Mean(),
        'standard_deviation': StdDev(),
        'cvar_50': CVaR(0.5),
        'cvar_95': CVaR(0.95),
        'cvar_99': CVaR(0.99),
        'worst_case': WorstCase(),
        'mae': MAE() 
    }
    # Collect all available statistic names for error messaging
    stats_instances_list = []
    available_stats = ', '.join(stats_map.keys())

    stats_instances_list = []
    for stat in statistics:
        stat_i = stats_map.get(stat.lower())
        if stat_i is None:
                f"Unrecognized statistic '{stat}'. Available statistics are: {available_stats}."
        stats_instances_list.append(stat_i)

    print(stats_instances_list)
    return stats_instances_list

def load_agent(agent_name, model_name, models_dir, instrument, contingent_claim, bump_size, path_transformation_configs, n_hedging_timesteps):
    # Initialize agent with additional parameters if necessary
    if agent_name in ['GeometricAsianNumericalDeltaHedgingAgent', 'ArithmeticAsianMonteCarloAgent', 'ArithmeticAsianControlVariateAgent']:
        agent = get_agent(agent_name, instrument, contingent_claim, bump_size=bump_size)
    else:
        agent = get_agent(agent_name, instrument, contingent_claim, path_transformation_configs=path_transformation_configs, n_hedging_timesteps=n_hedging_timesteps)
    
    # Define model path
    model_path = os.path.join(models_dir, agent.name, f'{model_name}.keras')
    
    # Load model if exists
    if agent.is_trainable:

        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"Loaded model for {agent.name} from {model_path}")
        else:
            print(f"No existing model found for {agent.name} at {model_path}. Ensure the model is trained before evaluation.")
    
    return agent

def main():
    args = parse_arguments()
    agent_model_names = parse_agent_model_names(args.agent_model_names)

    # Set up instruments
    instrument1 = GBMStock(S0=args.S0, T=args.T, N=args.N, r=args.r, sigma=args.sigma)
    instruments = [instrument1]

    # Define contingent claim
    contingent_claim = get_contingent_claim(args.contingent_claim, strike=args.strike)

    # Path transformation configurations
    transformation_type = 'log_moneyness'

    path_transformation_configs = [
        {'transformation_type': transformation_type, 'K': contingent_claim.strike}
    ]

    # Cost function
    cost_function = ProportionalCost(proportion=args.proportional_cost)

    # Risk measure
    risk_measure = CVaR(alpha=args.cvar_alpha)

    # Initialize agents with agent-specific model names
    agents = []
    for agent_name in args.agents:
        # Get model_name for this agent
        if agent_model_names and agent_name in agent_model_names:
            model_name = agent_model_names[agent_name]
        else:
            model_name = args.model_name  # default model_name

        agent = load_agent(
            agent_name=agent_name,
            model_name=model_name,
            models_dir=args.models_dir,
            instrument=instruments[0],
            contingent_claim=contingent_claim,
            bump_size=args.bump_size,
            path_transformation_configs=path_transformation_configs,
            n_hedging_timesteps=args.N
        )
        agents.append(agent)

    # Define optimizer paths with agent-specific model names
    optimizer_paths = {}
    for agent in agents:
        # Get model_name for this agent
        if agent_model_names and agent.name in agent_model_names:
            model_name = agent_model_names[agent.name]
        else:
            model_name = args.model_name  # default model_name

        optimizer_path = os.path.join(args.optimizers_dir, agent.name, model_name)
        optimizer_paths[agent.name] = optimizer_path

    # Initialize environment with the first agent as primary
    primary_agent = agents[0]
    env = Environment(
        agent=primary_agent,  # Primary agent for environment setup
        T=args.T,
        N=args.N,
        r=args.r,
        instrument_list=instruments,
        n_instruments=1,
        contingent_claim=contingent_claim,
        cost_function=cost_function,
        risk_measure=risk_measure,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=tf.keras.optimizers.Adam
    )

    print(f"Environment initialized with primary agent: {primary_agent.name}")
    print(agents)

    # Define directories for saving plots and stats
    os.makedirs(args.save_plots_dir, exist_ok=True)
    os.makedirs(args.save_stats_dir, exist_ok=True)
    if args.plot_histograms:
        os.makedirs(args.bootstrap_plots_dir, exist_ok=True)

    # Perform bootstrap confidence intervals for all agents
    print('Statistics:', args.statistics)
    stats_instances = get_statistics(args.statistics)
    print("Computing bootstrap confidence intervals...")
    df = env.bootstrap_confidence_intervals(
        agents=agents,
        statistics=stats_instances,
        n_paths=args.n_paths,
        n_bootstraps=args.n_bootstraps,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
        plot_histograms=args.plot_histograms,
        save_plot_dir=args.bootstrap_plots_dir,
        language=args.language,
        save_actions_path=args.save_actions_path,
        fixed_actions_paths=parse_fixed_actions_paths(args.fixed_actions_paths),
        pricing_method=args.pricing_method
    )
    agent_names = [agent.name for agent in agents]
    stat_names = [stat.name for stat in stats_instances]
    # Save the results
    results_path = os.path.join(args.save_stats_dir, f'bootstrap_{'_'.join(agent_names)}_{'_'.join(stat_names)}.xlsx')
    df.to_excel(results_path, index=False)
    print(f"Bootstrap statistics results saved to {results_path}")

    print("All evaluations completed.")

if __name__ == "__main__":
    main()
