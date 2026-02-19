import os
import time

import tensorflow as tf

from DeepHedging.Agents import (
    GeometricAsianDeltaHedgingAgent,
    GeometricAsianNumericalDeltaHedgingAgent,
)
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.ContingentClaims import AsianGeometricCall
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, Entropy, WorstCase
from DeepHedging.Environments import Environment


T = 22 / 365
N = 22
r = 0.05
n_instruments = 1

instrument = GBMStock(S0=100, T=T, N=N, r=r, sigma=0.05)
instruments = [instrument]
contingent_claim = AsianGeometricCall(strike=100)

cost_function = ProportionalCost(proportion=0.0)
risk_measure = CVaR(alpha=0.5)

delta_agent = GeometricAsianDeltaHedgingAgent(instrument, contingent_claim)
numerical_delta_agent = GeometricAsianNumericalDeltaHedgingAgent(instrument, contingent_claim, bump_size=0.001)

print(f"Analytical price: {delta_agent.get_model_price()}")
print(f"Numerical price: {numerical_delta_agent.get_model_price(instrument.S0, T)}")

env = Environment(
    agent=delta_agent,
    T=T,
    N=N,
    r=r,
    instrument_list=instruments,
    n_instruments=n_instruments,
    contingent_claim=contingent_claim,
    cost_function=cost_function,
    risk_measure=risk_measure,
    n_epochs=1,
    batch_size=2_000,
    learning_rate=0.001,
    optimizer=tf.keras.optimizers.Adam
)

print(time.ctime())

measures = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase(), Entropy()]

q = env.terminal_hedging_error_multiple_agents(
    agents=[delta_agent, numerical_delta_agent],
    n_paths=100_000,
    random_seed=33,
    plot_error=True,
    colors=["orange", "steelblue"],
    loss_functions=measures,
    plot_title="Error de Cobertura Terminal",
    save_plot_path=os.path.join(
        os.getcwd(), "assets", "plots", f"asian_geometric_{delta_agent.name}_comparison.pdf"
    ),
    save_stats_path=os.path.join(
        os.getcwd(), "assets", "csvs", f"asian_geometric_{delta_agent.name}_comparison.xlsx"
    ),
    min_x=-2,
    max_x=2,
    language="es",
)
print(q)

print(time.ctime())
