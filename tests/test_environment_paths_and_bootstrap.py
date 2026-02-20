import numpy as np
import tensorflow as tf

from DeepHedging.Agents import DeltaHedgingAgent
from DeepHedging.ContingentClaims import EuropeanCall
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.Environments import Environment
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.RiskMeasures import CVaR



def _build_env():
    stock = GBMStock(S0=100.0, T=2 / 252, N=2, r=0.01, sigma=0.2)
    claim = EuropeanCall(strike=100.0)
    agent = DeltaHedgingAgent(stock, claim)
    env = Environment(
        agent=agent,
        T=stock.T,
        N=stock.N,
        r=stock.r,
        instrument_list=[stock],
        n_instruments=1,
        contingent_claim=claim,
        cost_function=ProportionalCost(0.0),
        risk_measure=CVaR(0.5),
        n_epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        optimizer=tf.keras.optimizers.Adam,
    )
    return env, agent



def test_terminal_eval_with_paths_to_test_and_per_path_r():
    env, agent = _build_env()
    paths_2d = np.array(
        [
            [100.0, 101.0, 102.0],
            [100.0, 99.0, 98.0],
            [100.0, 100.0, 100.0],
            [100.0, 103.0, 101.0],
        ],
        dtype=np.float32,
    )
    paths = np.expand_dims(paths_2d, axis=-1)
    per_path_r = np.array([0.01, 0.02, 0.015, 0.005], dtype=np.float32)

    payload, errors = env.terminal_hedging_error_multiple_agents(
        agents=[agent],
        n_paths=4,
        random_seed=1,
        paths_to_test=paths,
        per_path_r=per_path_r,
        plot_error=False,
        loss_functions=None,
        return_errors=True,
    )
    mean_errors, std_errors, _ = payload
    assert len(mean_errors) == 1
    assert len(std_errors) == 1
    assert len(errors) == 1
    assert errors[0].shape[0] == 4



def test_bootstrap_with_moving_block_paths_to_test():
    env, agent = _build_env()
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(40, 2)), axis=1))
    prices = np.concatenate([100.0 * np.ones((40, 1)), prices], axis=1)
    paths = np.expand_dims(prices.astype(np.float32), axis=-1)

    df = env.bootstrap_confidence_intervals(
        agents=[agent],
        statistics=[CVaR(0.5)],
        n_paths=40,
        n_bootstraps=50,
        confidence_level=0.95,
        random_seed=11,
        paths_to_test=paths,
        bootstrap_method="moving_block",
        moving_block_size=4,
        batch_size=10,
        plot_histograms=False,
        pricing_method="fixed",
    )
    assert not df.empty
    assert "CVaR_50_point_estimate" in df.columns
