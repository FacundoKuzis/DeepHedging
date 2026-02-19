import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings
import pandas as pd
import inspect

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
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase
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
                agent_name, model_name = item.split('=', 1)
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
    Parses a list of strings in the format 'agent_name=path/to/actions.npy' into a dictionary.
    """
    fixed_paths = {}
    if arg_list:
        for item in arg_list:
            try:
                agent_name, path = item.split('=', 1)
                fixed_paths[agent_name] = path
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid format for fixed_actions_paths: '{item}'. "
                    "Expected format 'agent_name=path/to/actions.npy'"
                )
    if fixed_paths == {}:
        fixed_paths = None
    return fixed_paths

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Deep Hedging agents with specified parameters.")

    # Simulation parameters
    parser.add_argument('--T', type=float, default=None, help='Time to maturity in years. If omitted, uses N / trading_days_per_year.')
    parser.add_argument('--N', type=int, default=22, help='Number of time steps (default: 22)')
    parser.add_argument('--trading_days_per_year', type=int, default=252, help='Trading-day convention (default: 252)')
    parser.add_argument('--r', type=float, default=0.05, help='Risk-free rate (default: 0.05)')
    parser.add_argument('--S0', type=float, default=100, help='Initial stock price (default: 100)')
    parser.add_argument('--sigma', type=float, default=0.05, help='Volatility (default: 0.05)')
    parser.add_argument('--strike', type=float, default=100, help='Strike price (default: 100)')
    parser.add_argument('--claim_underlying_index', type=int, default=0, help='Underlying instrument index used by the claim (default: 0)')

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
    parser.add_argument('--benchmark_num_simulations', type=int, default=10000, help='MC paths for benchmark agents (default: 10000)')
    parser.add_argument('--benchmark_seed', type=int, default=33, help='MC seed for benchmark agents (default: 33)')
    parser.add_argument('--no_trade_band', type=float, default=0.0, help='No-trade band eta for benchmark agents (default: 0.0)')

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
    parser.add_argument('--min_x', type=float, default=-2.0, help='Minimum x-axis value for plots (default: -2.0)')
    parser.add_argument('--max_x', type=float, default=2.0, help='Maximum x-axis value for plots (default: 2.0)')
    parser.add_argument('--language', type=str, choices=['en', 'es'], default='es', help='Language for labels: "en" for English or "es" for Spanish (default: es)')
    parser.add_argument('--plot_title', type=str, default=None, help='Title for comparison plot')

    # File paths
    parser.add_argument('--model_name', type=str, default='asian_1', help='Model name (default: asian_1)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to load models from (default: models)')
    parser.add_argument('--optimizers_dir', type=str, default='optimizers', help='Directory to load optimizers from (default: optimizers)')
    parser.add_argument('--save_plots_dir', type=str, default='assets/plots', help='Directory to save plots (default: assets/plots)')
    parser.add_argument('--save_stats_dir', type=str, default='assets/csvs', help='Directory to save statistics (default: assets/csvs)')
    parser.add_argument('--save_actions_path', type=str, default=None, 
                        help='Directory to save each agent\'s actions as NPY files (default: None)')
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

    return parser.parse_args()

def get_agent(
    agent_name,
    instrument,
    contingent_claim,
    path_transformation_configs=None,
    n_hedging_timesteps=None,
    bump_size=0.001,
    benchmark_num_simulations=10000,
    benchmark_seed=33,
    no_trade_band=0.0,
):

    if agent_name not in AGENTS:
        raise ValueError(f"Agent '{agent_name}' is not recognized. Available agents: {list(AGENTS.keys())}")
    agent_class = AGENTS[agent_name]
    if agent_class.is_trainable:
        path_transformation_configs = [{'transformation_type': 'log_moneyness', 'K': contingent_claim.strike}]
        init_signature = inspect.signature(agent_class.__init__)
        init_params = init_signature.parameters
        agent_kwargs = {"path_transformation_configs": path_transformation_configs}
        if "n_hedging_timesteps" in init_params:
            agent_kwargs["n_hedging_timesteps"] = n_hedging_timesteps
        if "n_instruments" in init_params:
            agent_kwargs["n_instruments"] = 1
        return agent_class(**agent_kwargs)
    init_signature = inspect.signature(agent_class.__init__)
    init_params = init_signature.parameters
    candidate_kwargs = {
        "bump_size": bump_size,
        "num_simulations": benchmark_num_simulations,
        "seed": benchmark_seed,
        "no_trade_band": no_trade_band,
    }
    agent_kwargs = {k: v for k, v in candidate_kwargs.items() if k in init_params}
    return agent_class(instrument, contingent_claim, **agent_kwargs)

def get_contingent_claim(claim_type, strike, underlying_index=0):
    claims = {
        'EuropeanCall': EuropeanCall(strike=strike, underlying_index=underlying_index),
        'EuropeanPut': EuropeanPut(strike=strike, underlying_index=underlying_index),
        'AsianGeometricCall': AsianGeometricCall(strike=strike, underlying_index=underlying_index),
        'AsianGeometricPut': AsianGeometricPut(strike=strike, underlying_index=underlying_index),
        'AsianArithmeticCall': AsianArithmeticCall(strike=strike, underlying_index=underlying_index),
        'AsianArithmeticPut': AsianArithmeticPut(strike=strike, underlying_index=underlying_index)
    }
    if claim_type not in claims:
        raise ValueError(f"Contingent Claim '{claim_type}' is not recognized. Available claims: {list(claims.keys())}")
    return claims[claim_type]

def load_agent(
    agent_name,
    model_name,
    models_dir,
    instrument,
    contingent_claim,
    bump_size,
    path_transformation_configs,
    n_hedging_timesteps,
    benchmark_num_simulations,
    benchmark_seed,
    no_trade_band,
):
    # Initialize agent with additional parameters if necessary
    agent = get_agent(
        agent_name,
        instrument,
        contingent_claim,
        path_transformation_configs=path_transformation_configs,
        n_hedging_timesteps=n_hedging_timesteps,
        bump_size=bump_size,
        benchmark_num_simulations=benchmark_num_simulations,
        benchmark_seed=benchmark_seed,
        no_trade_band=no_trade_band,
    )
    
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
    if args.T is None:
        args.T = args.N / float(args.trading_days_per_year)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)
    agent_model_names = parse_agent_model_names(args.agent_model_names)

    # Set up instruments
    instrument1 = GBMStock(S0=args.S0, T=args.T, N=args.N, r=args.r, sigma=args.sigma)
    instruments = [instrument1]

    # Define contingent claim
    contingent_claim = get_contingent_claim(
        args.contingent_claim,
        strike=args.strike,
        underlying_index=args.claim_underlying_index,
    )

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
    agent_model_names_resolved = []
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
            n_hedging_timesteps=args.N,
            benchmark_num_simulations=args.benchmark_num_simulations,
            benchmark_seed=args.benchmark_seed,
            no_trade_band=args.no_trade_band,
        )
        agents.append(agent)
        agent_model_names_resolved.append(model_name)

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

    # Define measures
    measures = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase()]

    # Define directories for saving plots and stats
    os.makedirs(args.save_plots_dir, exist_ok=True)
    os.makedirs(args.save_stats_dir, exist_ok=True)

    # Compare primary agent with other agents
    for comparison_agent in agents[1:]:
        print(f"Evaluating {comparison_agent.name} against {primary_agent.name}")
        
        # Get colors from the agent attributes or set default
        try:
            primary_color = primary_agent.plot_color
        except AttributeError:
            primary_color = 'blue'

        try:
            comparison_color = comparison_agent.plot_color
        except AttributeError:
            comparison_color = 'orange'

        plot_title = {
            'en': 'Terminal Hedging Error',
            'es': 'Error de Cobertura Final'
        }

        # Get model names for the agents
        primary_model_name = agent_model_names_resolved[0]
        comparison_model_name = agent_model_names_resolved[agents.index(comparison_agent)]

        # Use model names in file paths
        save_plot_path = os.path.join(
            args.save_plots_dir,
            f'{primary_agent.name}_{primary_model_name}_vs_{comparison_agent.name}_{comparison_model_name}_comparison.pdf'
        )
        save_stats_path = os.path.join(
            args.save_stats_dir,
            f'{primary_agent.name}_{primary_model_name}_vs_{comparison_agent.name}_{comparison_model_name}_comparison.xlsx'
        )

        q = env.terminal_hedging_error_multiple_agents(
            agents=[primary_agent, comparison_agent], 
            n_paths=args.n_paths, 
            random_seed=args.random_seed, 
            plot_error=True, 
            colors=[primary_color, comparison_color],  
            loss_functions=measures, 
            plot_title=args.plot_title if args.plot_title else plot_title.get(args.language),
            save_plot_path=save_plot_path,
            save_stats_path=save_stats_path,
            min_x=args.min_x, 
            max_x=args.max_x,
            language=args.language,
            save_actions_path=args.save_actions_path,
            fixed_actions_paths=parse_fixed_actions_paths(args.fixed_actions_paths),
            pricing_method=args.pricing_method
        )
        print(f"Evaluation result for {comparison_agent.name}: {q}")

    print("All evaluations completed.")

if __name__ == "__main__":
    main()
