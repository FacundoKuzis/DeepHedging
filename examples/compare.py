import tensorflow as tf
import os
from DeepHedging.Agents import SimpleAgent, RecurrentAgent, LSTMAgent, DeltaHedgingAgent, WaveNetAgent, GRUAgent, \
    AsianDeltaHedgingAgent, QuantlibAsianGeometricAgent, QuantlibAsianArithmeticAgent
from DeepHedging.HedgingInstruments import GBMStock, HestonStock
from DeepHedging.ContingentClaims import EuropeanCall, AsianGeometricCall, AsianArithmeticCall, AsianArithmeticPut 
#from DeepHedging.ContingentClaims.asian_options import FloatingStrikeAsianGeometricCall as AsianGeometricCall 

from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase
from DeepHedging.Environments import Environment
import time

T = 22/365
N = 22
r = 0.0
n_instruments = 1

instrument1 = GBMStock(S0=1, T=T, N=N, r=r, sigma=0.2)

instruments = [instrument1] #instrument1
contingent_claim = AsianArithmeticCall(strike=1)

path_transformation_configs = [
    {'transformation_type': 'log_moneyness', 'K': contingent_claim.strike}#,
    #{'transformation_type': None}
]

cost_function = ProportionalCost(proportion=0.0)
risk_measure = CVaR(alpha=0.5)

delta_agent = AsianDeltaHedgingAgent(instrument1, contingent_claim)
bs_price = delta_agent.get_model_price()
print('p:', bs_price)
#bs_price = 0
agent = delta_agent

quantlib_agent = QuantlibAsianArithmeticAgent(instrument1, contingent_claim)
print('p quant:', quantlib_agent.get_model_price())

model_name = 'asian_1'


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
    learning_rate=0.001,
    optimizer=tf.keras.optimizers.Adam
)

print(time.ctime())

#env.load_optimizer(optimizer_path, only_weights=True)

agent_delta = delta_agent

#agent_simple = SimpleAgent(path_transformation_configs=path_transformation_configs)
#agent_simple.load_model(os.path.join(os.getcwd(), 'models', agent_simple.name, f'{model_name}.keras'))

#agent_recurrent = RecurrentAgent(path_transformation_configs=path_transformation_configs)
#agent_recurrent.load_model(os.path.join(os.getcwd(), 'models', agent_recurrent.name, f'{model_name}.keras'))

#agent_lstm = LSTMAgent(N, path_transformation_configs=path_transformation_configs)
#agent_lstm.load_model(os.path.join(os.getcwd(), 'models', agent_lstm.name, f'{model_name}.keras'))

#agent_wavenet = WaveNetAgent(N, path_transformation_configs=path_transformation_configs)
#agent_wavenet.load_model(os.path.join(os.getcwd(), 'models', agent_wavenet.name, f'{model_name}.keras'))

#agent_gru = GRUAgent(N, path_transformation_configs=path_transformation_configs)
#agent_gru.load_model(os.path.join(os.getcwd(), 'models', agent_gru.name, f'{model_name}.keras'))

measures = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase(), Entropy()]

q = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, quantlib_agent], 
                                           n_paths=100, random_seed=33, plot_error=True, 
                                           colors = ['orange', 'steelblue'], loss_functions = measures, plot_title='Terminal Hedging Error',
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}f_{quantlib_agent.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{quantlib_agent.name}f_comparision.xlsx'))
print(q)




"""
b = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, delta_agent], 
                                           n_paths=10_000, random_seed=33, plot_error=True, 
                                           colors = ['orange', 'steelblue'], loss_functions = measures, plot_title='Terminal Hedging Error',
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}b_{agent_recurrent.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_recurrent.name}b_comparision.xlsx'))
print(b)


a = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, agent_lstm], 
                                           n_paths=10_000, random_seed=33, plot_error=True, 
                                           colors = ['orange', 'forestgreen'], loss_functions = measures, plot_title='Terminal Hedging Error',
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_lstm.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_lstm.name}_comparision.xlsx'))
print(a)

d = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, agent_wavenet], 
                                           n_paths=10_000, random_seed=33, plot_error=True, 
                                           colors = ['orange', 'pink'], loss_functions = measures, plot_title='Terminal Hedging Error',
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_wavenet.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_wavenet.name}_comparision.xlsx'))
print(d)


e = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, agent_gru], 
                                           n_paths=10_000, random_seed=33, plot_error=True, 
                                           colors = ['orange', 'gray'], loss_functions = measures,
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_gru.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_gru.name}_comparision.xlsx'))
print(e)


c = env.terminal_hedging_error_multiple_agents(agents=[agent_simple, agent_delta], 
                                           n_paths=10_000, random_seed=33, plot_error=True, 
                                           colors = ['mediumorchid', 'orange'], loss_functions = measures,
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_simple.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_simple.name}_comparision.xlsx'))

print(c)


"""
print(time.ctime())
