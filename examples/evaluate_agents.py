import argparse
import tensorflow as tf
import os
from DeepHedging.Agents import (BaseAgent, SimpleAgent, RecurrentAgent, LSTMAgent, 
                                GRUAgent, WaveNetAgent, DeltaHedgingAgent, 
                                GeometricAsianDeltaHedgingAgent, GeometricAsianDeltaHedgingAgent2, 
                                GeometricAsianNumericalDeltaHedgingAgent, QuantlibAsianGeometricAgent, 
                                ArithmeticAsianMonteCarloAgent)

from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.ContingentClaims import (
    EuropeanCall, EuropeanPut, AsianGeometricCall, AsianGeometricPut,
    AsianArithmeticCall, AsianArithmeticPut
)
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase
from DeepHedging.Environments import Environment


def get_agent(agent_name, instrument, contingent_claim, path_transformation_configs=None, n_hedging_timesteps=None, **kwargs):

    agents = {
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
        'ArithmeticAsianMonteCarloAgent': ArithmeticAsianMonteCarloAgent
    }

    if agent_name not in agents:
        raise ValueError(f"Agent '{agent_name}' is not recognized. Available agents: {list(agents.keys())}")
    agent_class = agents[agent_name]
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Deep Hedging agents with specified parameters.")

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
                        choices=[
                            'SimpleAgent', 'RecurrentAgent', 'LSTMAgent', 'DeltaHedgingAgent', 
                            'WaveNetAgent', 'GRUAgent', 'AsianDeltaHedgingAgent', 
                            'AsianDeltaHedgingAgent2', 'QuantlibAsianGeometricAgent', 
                            'QuantlibAsianArithmeticAgent', 'AsianNumericalDeltaHedgingAgent'
                        ],
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
    parser.add_argument('--min_x', type=float, default=-2.0, help='Minimum x-axis value for plots (default: -2.0)')
    parser.add_argument('--max_x', type=float, default=2.0, help='Maximum x-axis value for plots (default: 2.0)')
    parser.add_argument('--language', type=str, choices=['en', 'es'], default='es', help='Language for labels: "en" for English or "es" for Spanish (default: es)')

    # File paths
    parser.add_argument('--model_name', type=str, default='asian_1', help='Model name (default: asian_1)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to load models from (default: models)')
    parser.add_argument('--optimizers_dir', type=str, default='optimizers', help='Directory to load optimizers from (default: optimizers)')
    parser.add_argument('--save_plots_dir', type=str, default='assets/plots', help='Directory to save plots (default: assets/plots)')
    parser.add_argument('--save_stats_dir', type=str, default='assets/csvs', help='Directory to save statistics (default: assets/csvs)')
    parser.add_argument('--save_actions_path', type=str, default=None, 
                        help='Directory to save each agent\'s actions as CSV files (default: None)')
    parser.add_argument('--fixed_actions_paths', type=str, nargs='*', default=None,
                        help='Fixed actions paths in the format agent_name=path/to/actions.csv. '
                            'Example: agent1=path/to/agent1_actions.csv agent2=path/to/agent2_actions.csv')


    return parser.parse_args()

def parse_fixed_actions_paths(arg_list):
    """
    Parses a list of strings in the format 'agent_name=path/to/actions.csv' into a dictionary.

    Arguments:
    - arg_list (list of str): List of strings to parse.

    Returns:
    - dict: Dictionary mapping agent names to file paths.
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
    return fixed_paths



def load_agent(agent_name, model_name, models_dir, instrument, contingent_claim, bump_size, path_transformation_configs, n_hedging_timesteps):
    # Initialize agent with additional parameters if necessary
    if agent_name == 'AsianNumericalDeltaHedgingAgent':
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

    # Initialize agents
    agents = []
    for agent_name in args.agents:
        agent = load_agent(
            agent_name=agent_name,
            model_name=args.model_name,
            models_dir=args.models_dir,
            instrument=instruments[0],
            contingent_claim=contingent_claim,
            bump_size=args.bump_size,
            path_transformation_configs=path_transformation_configs,
            n_hedging_timesteps=args.N
        )
        agents.append(agent)

    # Define optimizer paths
    optimizer_paths = {}
    for agent in agents:
        optimizer_path = os.path.join(args.optimizers_dir, agent.name, args.model_name)
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
    # Load optimizer states if they exist
    for agent in agents:
        if agent.is_trainable:
            optimizer_path = optimizer_paths[agent.name]
            if os.path.exists(optimizer_path):
                env.load_optimizer(optimizer_path, only_weights=True)
                print(f"Loaded optimizer for {agent.name} from {optimizer_path}")
            else:
                print(f"No existing optimizer found for {agent.name} at {optimizer_path}, starting fresh.")

    # Define measures
    measures = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase(), Entropy()]

    # Define directories for saving plots and stats
    os.makedirs(args.save_plots_dir, exist_ok=True)
    os.makedirs(args.save_stats_dir, exist_ok=True)

    # Compare primary agent with other agents
    for comparison_agent in agents[1:]:
        print(f"Evaluating {comparison_agent.name} against {primary_agent.name}")
        
        # Get colors from the COLORS dictionary
        try:
            primary_color = primary_agent.plot_color
            comparison_color = comparison_agent.plot_color
        except KeyError as e:
            print(f"Color for agent '{e.args[0]}' not found. Using default color 'black'.")
            primary_color = 'black'
            comparison_color = 'black'

        plot_title = {
            'en': 'Terminal Hedging Error',
            'es': 'Error de Cobertura Final'
        }

        q = env.terminal_hedging_error_multiple_agents(
            agents=[primary_agent, comparison_agent], 
            n_paths=args.n_paths, 
            random_seed=args.random_seed, 
            plot_error=True, 
            colors=[primary_color, comparison_color],  
            loss_functions=measures, 
            plot_title=plot_title.get(args.language),
            fixed_price=primary_agent.get_model_price(),
            save_plot_path=os.path.join(
                args.save_plots_dir, 
                f'{args.model_name}_{comparison_agent.name}_comparison.pdf'
            ),
            save_stats_path=os.path.join(
                args.save_stats_dir, 
                f'{args.model_name}_{comparison_agent.name}_comparison.xlsx'
            ),
            min_x=args.min_x, 
            max_x=args.max_x,
            language=args.language,
            save_actions_path=args.save_actions_path,
            fixed_actions_paths=parse_fixed_actions_paths(args.fixed_actions_paths)
        )
        print(f"Evaluation result for {comparison_agent.name}: {q}")

    print("All evaluations completed.")

if __name__ == "__main__":
    main()
