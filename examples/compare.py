import tensorflow as tf
import os
from DeepHedging.Agents import SimpleAgent, RecurrentAgent, LSTMAgent, DeltaHedgingAgent
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.ContingentClaims import EuropeanCall
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR
from DeepHedging.Environments import Environment
import time

instrument = GBMStock(S0=100, T=50/252, N=50, r=0.05, sigma=0.2)
contingent_claim = EuropeanCall(strike=100)
cost_function = ProportionalCost(proportion=0.0)
risk_measure = CVaR(alpha=0.5)
#risk_measure = MAE()
#agent = LSTMAgent(instrument.N)
#agent = RecurrentAgent()
delta_agent = DeltaHedgingAgent(instrument, contingent_claim)
bs_price = delta_agent.get_model_price()
agent = delta_agent
#agent = RecurrentAgent(path_transformation_type='log_moneyness', K = contingent_claim.strike)
#agent = LSTMAgent(instrument.N, path_transformation_type='log_moneyness', K = contingent_claim.strike)

model_path = os.path.join(os.getcwd(), 'models', agent.name, 'logm_1c_cvar50.keras')
optimizer_path = os.path.join(os.getcwd(), 'optimizers', agent.name, 'logm_1c_cvar50')

env = Environment(
    agent=agent,
    instrument=instrument,
    contingent_claim=contingent_claim,
    cost_function=cost_function,
    risk_measure=risk_measure,
    n_epochs=200,
    batch_size=2_000,
    learning_rate=0.001,
    optimizer=tf.keras.optimizers.Adam
)

print(time.ctime())

env.load_model(model_path)
env.load_optimizer(optimizer_path, only_weights=True)

agent_delta = DeltaHedgingAgent(instrument, contingent_claim)

agent_simple = SimpleAgent(path_transformation_type='log_moneyness', K = contingent_claim.strike)
agent_simple.load_model(os.path.join(os.getcwd(), 'models', agent_simple.name, 'logm_0c_cvar50_double.keras'))

agent_recurrent = RecurrentAgent(path_transformation_type='log_moneyness', K = contingent_claim.strike)
agent_recurrent.load_model(os.path.join(os.getcwd(), 'models', agent_recurrent.name, 'logm_0c_cvar50_double.keras'))

agent_lstm = LSTMAgent(instrument.N, path_transformation_type='log_moneyness', K = contingent_claim.strike)
agent_lstm.load_model(os.path.join(os.getcwd(), 'models', agent_lstm.name, 'logm_0c_cvar50_double_relu_.keras'))

"""
b = env.terminal_hedging_error_multiple_agents(agents=[agent_recurrent, agent_delta], 
                                           n_paths=100_000, random_seed=33, plot_error=True, 
                                           colors = ['steelblue', 'orange'],
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', 'comparision_recurrent_3.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', 'comparision_recurrent_3.csv'))
print(b)
"""
a = env.terminal_hedging_error_multiple_agents(agents=[agent_lstm, agent_delta], 
                                           n_paths=100_000, random_seed=33, plot_error=True, 
                                           colors = ['forestgreen', 'orange'],
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', 'comparision_lstm_3_1relu.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', 'comparision_lstm_3_2relu.csv'))
print(a)
"""
c = env.terminal_hedging_error_multiple_agents(agents=[agent_simple, agent_delta], 
                                           n_paths=100_000, random_seed=33, plot_error=True, 
                                           colors = ['mediumorchid', 'orange'],
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', 'comparision_simple_3.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', 'comparision_simple_3.csv'))

print(c)
"""
print(time.ctime())
