#!/usr/bin/env python3
# compare_hedging_strategy.py

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings
import pandas as pd

# Import DeepHedging modules (Ensure these are available in your environment)
from DeepHedging.Agents import (
    BaseAgent, SimpleAgent, RecurrentAgent, LSTMAgent, GRUAgent, WaveNetAgent, 
    DeltaHedgingAgent, GeometricAsianDeltaHedgingAgent, GeometricAsianDeltaHedgingAgent2, 
    GeometricAsianNumericalDeltaHedgingAgent, QuantlibAsianGeometricAgent, 
    ArithmeticAsianMonteCarloAgent, ArithmeticAsianControlVariateAgent, MonteCarloAgent
)

from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.ContingentClaims import (
    EuropeanCall, EuropeanPut, AsianGeometricCall, AsianGeometricPut,
    AsianArithmeticCall, AsianArithmeticPut
)
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase
from DeepHedging.Environments import Environment  # Ensure your Environment class includes compare_hedging_strategy

# Define available agents
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare Hedging Strategies of Deep Hedging Agents.")

    # Simulation parameters
    parser.add_argument('--T', type=float, default=22/365, help='Time to maturity in years (default: 22/365)')
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
                        help='List of agents to compare')
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
    parser.add_argument('--min_x', type=float, default=-2.0, help='Minimum x-axis value for plots (default: -2.0)')
    parser.add_argument('--max_x', type=float, default=2.0, help='Maximum x-axis value for plots (default: 2.0)')
    parser.add_argument('--language', type=str, choices=['en', 'es'], default='es', help='Language for labels: "en" for English or "es" for Spanish (default: es)')
    parser.add_argument('--plot_title', type=str, default=None, help='Title for comparison plot')

    # File paths
    parser.add_argument('--model_name', type=str, default='asian_1', help='Model name (default: asian_1)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to load models from (default: models)')
    parser.add_argument('--save_plots_dir', type=str, default='assets/plots', help='Directory to save plots (default: assets/plots)')

    # New argument for agent-specific model names
    parser.add_argument('--agent_model_names', type=str, nargs='*', default=None,
                        help='Agent model names in the format agent_name=model_name. '
                             'Example: agent1=model1 agent2=model2')

    return parser.parse_args()

def get_agent(agent_name, instrument, contingent_claim, path_transformation_configs=None, n_hedging_timesteps=None, **kwargs):
    """
    Initializes and returns an agent instance based on the agent name and provided parameters.
    """
    if agent_name not in AGENTS:
        raise ValueError(f"Agent '{agent_name}' is not recognized. Available agents: {list(AGENTS.keys())}")
    agent_class = AGENTS[agent_name]
    if hasattr(agent_class, 'is_trainable') and agent_class.is_trainable:
        # Define default path transformation configurations for trainable agents
        path_transformation_configs = [{'transformation_type': 'log_moneyness', 'K': contingent_claim.strike}]
        return agent_class(path_transformation_configs=path_transformation_configs,
                          n_hedging_timesteps=n_hedging_timesteps)
    return agent_class(instrument, contingent_claim, **kwargs)

def get_contingent_claim(claim_type, strike):
    """
    Returns an instance of the specified contingent claim type.
    """
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

def load_agent(agent_name, model_name, models_dir, instrument, contingent_claim, bump_size, path_transformation_configs, n_hedging_timesteps):
    """
    Loads an agent with the specified model. If the model file exists, it loads the model weights.
    """
    # Initialize agent with additional parameters if necessary
    if agent_name in ['GeometricAsianNumericalDeltaHedgingAgent', 
                     'ArithmeticAsianMonteCarloAgent', 
                     'ArithmeticAsianControlVariateAgent']:
        agent = get_agent(agent_name, instrument, contingent_claim, bump_size=bump_size)
    else:
        agent = get_agent(agent_name, instrument, contingent_claim, 
                         path_transformation_configs=path_transformation_configs, 
                         n_hedging_timesteps=n_hedging_timesteps)
    
    # Define model path
    model_path = os.path.join(models_dir, agent.name, f'{model_name}.keras')
    
    # Load model if exists
    if hasattr(agent, 'is_trainable') and agent.is_trainable:
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
    print(f"Agents to compare: {[agent.name for agent in agents]}")

    # Prepare save plot path
    comparison_model_names = [agent_model_names.get(agent.name, args.model_name) if agent_model_names else args.model_name for agent in agents]
    agents_plot_names = '_vs_'.join([f"{agent.name}_{model_name}" for agent, model_name in zip(agents, comparison_model_names)])
    save_plot_path = os.path.join(
        args.save_plots_dir,
        f'hedging_comparison_{agents_plot_names}.pdf'
    )

    # Call the compare_hedging_strategy method
    env.compare_hedging_strategy(
        agents=agents,
        n_paths=1,  # Typically, one path is sufficient for visual comparison
        random_seed=args.random_seed,
        save_plot_path=save_plot_path,
        language=args.language
    )

    print(f"Hedging strategies comparison plot saved to {save_plot_path}")
    print("Hedging strategies comparison completed.")

if __name__ == "__main__":
    main()
