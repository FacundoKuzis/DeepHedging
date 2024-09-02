import tensorflow as tf
import os
from DeepHedging.Agents import SimpleAgent, RecurrentAgent, LSTMAgent, DeltaHedgingAgent
from DeepHedging.HedgingInstruments import GBMStock, HestonStock
from DeepHedging.ContingentClaims import EuropeanCall
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR
from DeepHedging.Environments import Environment
import time

T = 63/252
N = 63
r = 0.05
n_instruments = 2

instrument1 = GBMStock(S0=100, T=T, N=N, r=r, sigma=0.2)
instrument2 = HestonStock(S0=100, T=T, N=N, r=r, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, return_variance=True)

instruments = [instrument2] #instrument1
contingent_claim = EuropeanCall(strike=100)

path_transformation_configs = [
    {'transformation_type': 'log_moneyness', 'K': contingent_claim.strike},
    {'transformation_type': None}
]

cost_function = ProportionalCost(proportion=0.02)
risk_measure = CVaR(alpha=0.5)
#risk_measure = MAE()
#agent = LSTMAgent(instrument.N)
#agent = RecurrentAgent()
delta_agent = DeltaHedgingAgent(instrument1, contingent_claim)
bs_price = delta_agent.get_model_price()
agent = delta_agent
agent = RecurrentAgent(path_transformation_configs = path_transformation_configs, n_instruments=n_instruments)
agent = LSTMAgent(N, path_transformation_configs=path_transformation_configs, n_instruments=n_instruments)

model_path = os.path.join(os.getcwd(), 'models', agent.name, 'logm_2c_cvar50.keras')
optimizer_path = os.path.join(os.getcwd(), 'optimizers', agent.name, 'logm_2c_cvar50')


initial_learning_rate = 0.001
decay_steps = 10 
decay_rate = 0.99  
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True 
)

env = Environment(
    agent=agent,
    T = T,
    N = N,
    r = r,
    instrument_list=instruments,
    n_instruments = n_instruments,
    contingent_claim=contingent_claim,
    cost_function=cost_function,
    risk_measure=risk_measure,
    n_epochs=200,
    batch_size=2_000,
    learning_rate=learning_rate_schedule,
    optimizer=tf.keras.optimizers.Adam
)

print(time.ctime())

agent.load_model(model_path)
env.load_optimizer(optimizer_path, only_weights=True)

#env.train(train_paths=20_000, val_paths=2000)

#agent.save_model(model_path)
#env.save_optimizer(optimizer_path)

env.terminal_hedging_error(n_paths=5000, random_seed=33, plot_error=True, fixed_price = bs_price, n_paths_for_pricing = 50_000, 
         save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', agent.name, 'hedge_error_cv50_2c_bsprice.pdf'))

#for i in range(10,20):
#   env.plot_hedging_strategy(os.path.join(os.getcwd(), 'assets', 'plots', agent.name, f'plot_{i+1}.pdf'), random_seed = i + 1)

print(time.ctime())
