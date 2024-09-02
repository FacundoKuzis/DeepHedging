import tensorflow as tf
import os
from DeepHedging.Agents import SimpleAgent, RecurrentAgent, LSTMAgent, DeltaHedgingAgent, WaveNetAgent, GRUAgent
from DeepHedging.HedgingInstruments import GBMStock, HestonStock
from DeepHedging.ContingentClaims import EuropeanCall
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase
from DeepHedging.Environments import Environment
import time

T = 63/252
N = 63
r = 0.05
n_instruments = 1

instrument1 = GBMStock(S0=100, T=T, N=N, r=r, sigma=0.2)
instrument2 = HestonStock(S0=100, T=T, N=N, r=r, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, return_variance=True)

instruments = [instrument1] #instrument1
contingent_claim = EuropeanCall(strike=100)

path_transformation_configs = [
    {'transformation_type': 'log_moneyness', 'K': contingent_claim.strike}#,
    #{'transformation_type': None}
]

cost_function = ProportionalCost(proportion=0.02)
risk_measure = CVaR(alpha=0.5)

delta_agent = DeltaHedgingAgent(instrument1, contingent_claim)
bs_price = delta_agent.get_model_price()
print('p:', bs_price)
agent = delta_agent

model_name = '2'


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

agent_delta = DeltaHedgingAgent(instrument1, contingent_claim)

#agent_simple = SimpleAgent(path_transformation_configs=path_transformation_configs)
#agent_simple.load_model(os.path.join(os.getcwd(), 'models', agent_simple.name, f'{model_name}.keras'))

agent_recurrent = RecurrentAgent(path_transformation_configs=path_transformation_configs)
agent_recurrent.load_model(os.path.join(os.getcwd(), 'models', agent_recurrent.name, f'{model_name}.keras'))

agent_lstm = LSTMAgent(N, path_transformation_configs=path_transformation_configs)
agent_lstm.load_model(os.path.join(os.getcwd(), 'models', agent_lstm.name, f'{model_name}.keras'))

agent_wavenet = WaveNetAgent(N, path_transformation_configs=path_transformation_configs)
agent_wavenet.load_model(os.path.join(os.getcwd(), 'models', agent_wavenet.name, f'{model_name}.keras'))

agent_gru = GRUAgent(N, path_transformation_configs=path_transformation_configs)
agent_gru.load_model(os.path.join(os.getcwd(), 'models', agent_gru.name, f'{model_name}.keras'))

measures = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase(), Entropy()]

b = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, agent_recurrent], 
                                           n_paths=100_000, random_seed=34, plot_error=True, 
                                           colors = ['orange', 'steelblue'], loss_functions = measures,
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_recurrent.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_recurrent.name}_comparision.xlsx'))
print(b)

a = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, agent_lstm], 
                                           n_paths=100_000, random_seed=34, plot_error=True, 
                                           colors = ['orange', 'forestgreen'], loss_functions = measures,
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_lstm.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_lstm.name}_comparision.xlsx'))
print(a)

d = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, agent_wavenet], 
                                           n_paths=100_000, random_seed=34, plot_error=True, 
                                           colors = ['orange', 'pink'], loss_functions = measures,
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_wavenet.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_wavenet.name}_comparision.xlsx'))
print(d)

e = env.terminal_hedging_error_multiple_agents(agents=[agent_delta, agent_gru], 
                                           n_paths=100_000, random_seed=34, plot_error=True, 
                                           colors = ['orange', 'gray'], loss_functions = measures,
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_gru.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_gru.name}_comparision.xlsx'))
print(e)

"""
c = env.terminal_hedging_error_multiple_agents(agents=[agent_simple, agent_delta], 
                                           n_paths=100_000, random_seed=33, plot_error=True, 
                                           colors = ['mediumorchid', 'orange'], loss_functions = measures,
                                           fixed_price = bs_price, save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', f'{model_name}_{agent_simple.name}_comparision.pdf'),
                                           save_stats_path=os.path.join(os.getcwd(), 'assets', 'csvs', f'{model_name}_{agent_simple.name}_comparision.xlsx'))

print(c)


"""
print(time.ctime())
