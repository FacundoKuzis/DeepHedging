import tensorflow as tf
import os
from DeepHedging.Agents import SimpleAgent, RecurrentAgent, LSTMAgent, DeltaHedgingAgent
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.ContingentClaims import EuropeanCall
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE
from DeepHedging.Environments import Environment

instrument = GBMStock(S0=100, T=50/252, N=50, r=0.05, sigma=0.2)
contingent_claim = EuropeanCall(strike=100)
cost_function = ProportionalCost(proportion=0.0)
risk_measure = MAE()
#agent = LSTMAgent(instrument.N)
#agent = RecurrentAgent()
agent = DeltaHedgingAgent(instrument, contingent_claim)


model_path = os.path.join(os.getcwd(), 'models', agent.name, 'model_test.keras')
optimizer_path = os.path.join(os.getcwd(), 'optimizers', agent.name, 'optimizer_test')

env = Environment(
    agent=agent,
    instrument=instrument,
    contingent_claim=contingent_claim,
    cost_function=cost_function,
    risk_measure=risk_measure,
    n_epochs=200,
    batch_size=1000,
    learning_rate=0.0001,
    optimizer=tf.keras.optimizers.Adam
)

env.load_model(model_path)
env.load_optimizer(optimizer_path, only_weights=True)


env.test(n_paths=5000, random_seed=33, plot_pnl=True, 
         save_plot_path=os.path.join(os.getcwd(), 'assets', 'plots', agent.name, 'pnl.pdf'))

#env.train(train_paths=10000, val_paths=2000)

#env.save_model(model_path)
#env.save_optimizer(optimizer_path)

#for i in range(10):
#    env.plot_hedging_strategy(os.path.join(os.getcwd(), 'assets', 'plots', agent.name, f'plot_{i+1}.pdf'), random_seed = i + 1)
