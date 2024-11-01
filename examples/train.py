import argparse
import tensorflow as tf
import os
import time
from DeepHedging.Agents import (BaseAgent, SimpleAgent, RecurrentAgent, LSTMAgent, 
                                GRUAgent, WaveNetAgent, DeltaHedgingAgent, 
                                GeometricAsianDeltaHedgingAgent, GeometricAsianDeltaHedgingAgent2, 
                                GeometricAsianNumericalDeltaHedgingAgent, QuantlibAsianGeometricAgent, 
                                ArithmeticAsianMonteCarloAgent, ArithmeticAsianControlVariateAgent)

from DeepHedging.HedgingInstruments import GBMStock, HestonStock
from DeepHedging.ContingentClaims import (
    EuropeanCall, EuropeanPut, AsianGeometricCall, AsianGeometricPut,
    AsianArithmeticCall, AsianArithmeticPut
)
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase
from DeepHedging.Environments import Environment

def get_agent(agent_name, path_transformation_configs, n_hedging_timesteps):
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
        'ArithmeticAsianMonteCarloAgent': ArithmeticAsianMonteCarloAgent,
        'ArithmeticAsianControlVariateAgent': ArithmeticAsianControlVariateAgent
    }
    if agent_name not in agents:
        raise ValueError(f"Agent '{agent_name}' is not recognized. Available agents: {list(agents.keys())}")
    return agents[agent_name](n_hedging_timesteps=n_hedging_timesteps, path_transformation_configs=path_transformation_configs)

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
    parser = argparse.ArgumentParser(description="Train a Deep Hedging agent with specified parameters.")

    # Simulation parameters
    parser.add_argument('--T', type=float, default=63/252, help='Time to maturity (default: 63/252)')
    parser.add_argument('--N', type=int, default=63, help='Number of time steps (default: 63)')
    parser.add_argument('--r', type=float, default=0.05, help='Risk-free rate (default: 0.05)')
    parser.add_argument('--S0', type=float, default=100, help='Initial stock price (default: 100)')
    parser.add_argument('--sigma', type=float, default=0.2, help='Volatility (default: 0.2)')
    parser.add_argument('--strike', type=float, default=100, help='Strike price (default: 100)')

    # Model parameters
    parser.add_argument('--agent', type=str, default='RecurrentAgent', 
                        choices=[
                            'SimpleAgent', 'RecurrentAgent', 'LSTMAgent', 'GRUAgent', 'WaveNetAgent', 
                            'DeltaHedgingAgent', 'GeometricAsianDeltaHedgingAgent', 'GeometricAsianDeltaHedgingAgent2', 
                            'GeometricAsianNumericalDeltaHedgingAgent', 'QuantlibAsianGeometricAgent', 
                            'ArithmeticAsianMonteCarloAgent', 'ArithmeticAsianControlVariateAgent'
                        ],
                        help='Type of agent to train (default: RecurrentAgent)')
    
    # Contingent claim parameters
    parser.add_argument('--contingent_claim', type=str, default='AsianGeometricCall',
                        choices=[
                            'EuropeanCall', 'EuropeanPut', 'AsianGeometricCall', 
                            'AsianGeometricPut', 'AsianArithmeticCall', 'AsianArithmeticPut'
                        ],
                        help='Type of contingent claim (default: AsianGeometricCall)')

    # Learning parameters
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
    parser.add_argument('--decay_steps', type=int, default=10, help='Decay steps for learning rate (default: 10)')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='Decay rate for learning rate (default: 0.99)')
    
    # Environment parameters
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size (default: 10000)')
    parser.add_argument('--train_paths', type=int, default=100000, help='Number of training paths (default: 100000)')
    
    # Cost function parameters
    parser.add_argument('--proportional_cost', type=float, default=0.0, help='Proportional cost (default: 0.0)')
    
    # Risk measure parameters
    parser.add_argument('--cvar_alpha', type=float, default=0.5, help='CVaR alpha (default: 0.5)')
    
    # Model paths
    parser.add_argument('--model_name', type=str, default='asian_1', help='Model name (default: asian_1)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save models (default: models)')
    parser.add_argument('--optimizers_dir', type=str, default='optimizers', help='Directory to save optimizers (default: optimizers)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set up instruments
    instrument1 = GBMStock(S0=args.S0, T=args.T, N=args.N, r=args.r, sigma=args.sigma)
    instruments = [instrument1]
    
    # Define contingent claim
    contingent_claim = get_contingent_claim(args.contingent_claim, strike=args.strike)
      
    path_transformation_configs = [
        {'transformation_type': "log_moneyness", 'K': contingent_claim.strike}
    ]
    
    # Cost function
    cost_function = ProportionalCost(proportion=args.proportional_cost)
    
    # Risk measure
    risk_measure = CVaR(alpha=args.cvar_alpha)
    
    # Initialize agent
    agent = get_agent(args.agent, path_transformation_configs, args.N)
    
    # Define model and optimizer paths
    model_path = os.path.join(os.getcwd(), args.models_dir, agent.name, f'{args.model_name}.keras')
    optimizer_path = os.path.join(os.getcwd(), args.optimizers_dir, agent.name, args.model_name)
    
    print(f"Training Agent: {agent.name}, Model Name: {args.model_name}")
    
    # Learning rate schedule
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.initial_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True 
    )
    
    # Initialize environment
    env = Environment(
        agent=agent,
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
        learning_rate=learning_rate_schedule,
        optimizer=tf.keras.optimizers.Adam
    )
    
    print(f"Training started at: {time.ctime()}")
    
    # Load existing model and optimizer if they exist
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"No existing model found at {model_path}, starting fresh.")
    
    if os.path.exists(optimizer_path):
        env.load_optimizer(optimizer_path, only_weights=True)
        print(f"Loaded optimizer from {optimizer_path}")
    else:
        print(f"No existing optimizer found at {optimizer_path}, starting fresh.")
    
    # Train the environment
    env.train(train_paths=args.train_paths)
    
    # Save the model and optimizer
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
    env.save_optimizer(optimizer_path)
    print(f"Optimizer saved to {optimizer_path}")
    
    print(f"Training completed at: {time.ctime()}")

if __name__ == "__main__":
    main()

"""
Use example:
python train.py --agent LSTMAgent --contingent_claim AsianArithmeticCall --T 0.25 --N 63 --r 0.05 --S0 100 --sigma 0.2 --strike 100 --initial_lr 0.001 --decay_steps 10 --decay_rate 0.99 --n_epochs 100 --batch_size 10000 --train_paths 100000 --proportional_cost 0.0 --cvar_alpha 0.5 --model_name asian_1 --models_dir models --optimizers_dir optimizers
"""
