"""
Single-file runner configured via global variables.

Usage:
    python examples/run_with_globals.py
"""

import os
import sys
import inspect
import random

import numpy as np

# Determinism flags must be set before importing TensorFlow.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tensorflow as tf

# Ensure local imports work without installing the package.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DeepHedging.Agents import (
    SimpleAgent, RecurrentAgent, LSTMAgent, GRUAgent, WaveNetAgent,
    DeltaHedgingAgent, GeometricAsianDeltaHedgingAgent, GeometricAsianDeltaHedgingAgent2,
    GeometricAsianNumericalDeltaHedgingAgent, QuantlibAsianGeometricAgent,
    ArithmeticAsianMonteCarloAgent, ArithmeticAsianControlVariateAgent, MonteCarloAgent,
)
from DeepHedging.HedgingInstruments import GBMStock
from DeepHedging.ContingentClaims import (
    EuropeanCall, EuropeanPut, AsianGeometricCall, AsianGeometricPut,
    AsianArithmeticCall, AsianArithmeticPut,
)
from DeepHedging.CostFunctions import ProportionalCost
from DeepHedging.RiskMeasures import MAE, CVaR, WorstCase
from DeepHedging.Environments import Environment


# ===========================
# Global Config (edit here)
# ===========================
RUN_MODE = "train_and_evaluate"  # "train", "evaluate", "train_and_evaluate"

# Market + contract
N = 63
TRADING_DAYS_PER_YEAR = 252
T = N / TRADING_DAYS_PER_YEAR
R = 0.05
S0 = 100.0
SIGMA = 0.2
STRIKE = 100.0
CLAIM_UNDERLYING_INDEX = 0
GLOBAL_RANDOM_SEED = 42

# Claim/risk
CONTINGENT_CLAIM_NAME = "AsianGeometricCall"
PROPORTIONAL_COST = 0.0
CVaR_ALPHA = 0.5

# Main agent
MAIN_AGENT_NAME = "LSTMAgent"
BUMP_SIZE = 0.001  # Used by numerical agents

# Train settings (fast defaults)
N_EPOCHS = 3
BATCH_SIZE = 1_000
TRAIN_PATHS = 4_000
VAL_PATHS = 1_000
INITIAL_LR = 0.001
DECAY_STEPS = 10
DECAY_RATE = 0.99
RESAMPLE_EACH_EPOCH = False

# Evaluate settings (fast defaults)
COMPARE_AGENT_NAMES = [
    "GeometricAsianDeltaHedgingAgent",
]
EVAL_PATHS = 4_000
EVAL_SEED = 33
PLOT_MIN_X = -2.0
PLOT_MAX_X = 2.0
LANGUAGE = "en"
PRICING_METHOD = "fixed"  # "fixed" or "individual"

# Persistence
MODEL_NAME = "quick_run"
MODELS_DIR = "models"
OPTIMIZERS_DIR = "optimizers"
PLOTS_DIR = os.path.join("assets", "plots")
STATS_DIR = os.path.join("assets", "csvs")
LOAD_IF_EXISTS = True
SAVE_AFTER_TRAIN = True

# Optional model-name overrides for evaluation agents (key = class name)
AGENT_MODEL_NAME_OVERRIDES = {
    # "LSTMAgent": "another_model_name",
}


AGENTS = {
    "SimpleAgent": SimpleAgent,
    "RecurrentAgent": RecurrentAgent,
    "LSTMAgent": LSTMAgent,
    "GRUAgent": GRUAgent,
    "WaveNetAgent": WaveNetAgent,
    "DeltaHedgingAgent": DeltaHedgingAgent,
    "GeometricAsianDeltaHedgingAgent": GeometricAsianDeltaHedgingAgent,
    "GeometricAsianDeltaHedgingAgent2": GeometricAsianDeltaHedgingAgent2,
    "GeometricAsianNumericalDeltaHedgingAgent": GeometricAsianNumericalDeltaHedgingAgent,
    "QuantlibAsianGeometricAgent": QuantlibAsianGeometricAgent,
    "ArithmeticAsianMonteCarloAgent": ArithmeticAsianMonteCarloAgent,
    "ArithmeticAsianControlVariateAgent": ArithmeticAsianControlVariateAgent,
    "MonteCarloAgent": MonteCarloAgent,
}

CLAIMS = {
    "EuropeanCall": EuropeanCall,
    "EuropeanPut": EuropeanPut,
    "AsianGeometricCall": AsianGeometricCall,
    "AsianGeometricPut": AsianGeometricPut,
    "AsianArithmeticCall": AsianArithmeticCall,
    "AsianArithmeticPut": AsianArithmeticPut,
}


def build_claim():
    if CONTINGENT_CLAIM_NAME not in CLAIMS:
        raise ValueError(f"Unknown claim '{CONTINGENT_CLAIM_NAME}'. Available: {list(CLAIMS.keys())}")
    return CLAIMS[CONTINGENT_CLAIM_NAME](strike=STRIKE, underlying_index=CLAIM_UNDERLYING_INDEX)


def build_instrument():
    return GBMStock(S0=S0, T=T, N=N, r=R, sigma=SIGMA)


def build_agent(agent_name, instrument, contingent_claim):
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent '{agent_name}'. Available: {list(AGENTS.keys())}")

    agent_class = AGENTS[agent_name]
    path_transformation_configs = [{"transformation_type": "log_moneyness", "K": contingent_claim.strike}]

    if agent_class.is_trainable:
        init_signature = inspect.signature(agent_class.__init__)
        init_params = init_signature.parameters
        kwargs = {"path_transformation_configs": path_transformation_configs}
        if "n_hedging_timesteps" in init_params:
            kwargs["n_hedging_timesteps"] = N
        if "n_instruments" in init_params:
            kwargs["n_instruments"] = 1
        return agent_class(**kwargs)

    kwargs = {}
    if agent_name in {
        "GeometricAsianNumericalDeltaHedgingAgent",
        "ArithmeticAsianMonteCarloAgent",
        "ArithmeticAsianControlVariateAgent",
        "MonteCarloAgent",
    }:
        kwargs["bump_size"] = BUMP_SIZE

    return agent_class(instrument, contingent_claim, **kwargs)


def get_model_name_for_agent(agent_name):
    return AGENT_MODEL_NAME_OVERRIDES.get(agent_name, MODEL_NAME)


def get_agent_paths(agent, agent_name):
    model_name = get_model_name_for_agent(agent_name)
    model_path = os.path.join(ROOT_DIR, MODELS_DIR, agent.name, f"{model_name}.keras")
    optimizer_path = os.path.join(ROOT_DIR, OPTIMIZERS_DIR, agent.name, model_name)
    return model_path, optimizer_path


def maybe_load_trainable_agent(agent, agent_name, env):
    if not agent.is_trainable or not LOAD_IF_EXISTS:
        return

    model_path, optimizer_path = get_agent_paths(agent, agent_name)
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"[load] model: {model_path}")
    if os.path.exists(optimizer_path):
        env.load_optimizer(optimizer_path, only_weights=True)
        print(f"[load] optimizer: {optimizer_path}")


def maybe_save_trainable_agent(agent, agent_name, env):
    if not agent.is_trainable or not SAVE_AFTER_TRAIN:
        return

    model_path, optimizer_path = get_agent_paths(agent, agent_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
    agent.save_model(model_path)
    env.save_optimizer(optimizer_path)
    print(f"[save] model: {model_path}")
    print(f"[save] optimizer: {optimizer_path}")


def main():
    run_mode_values = {"train", "evaluate", "train_and_evaluate"}
    if RUN_MODE not in run_mode_values:
        raise ValueError(f"Invalid RUN_MODE '{RUN_MODE}'. Use one of {run_mode_values}.")

    # Best-effort deterministic execution on CPU.
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    if GLOBAL_RANDOM_SEED is not None:
        random.seed(GLOBAL_RANDOM_SEED)
        np.random.seed(GLOBAL_RANDOM_SEED)
        tf.random.set_seed(GLOBAL_RANDOM_SEED)
        try:
            tf.keras.utils.set_random_seed(GLOBAL_RANDOM_SEED)
        except Exception:
            pass

    instrument = build_instrument()
    instruments = [instrument]
    contingent_claim = build_claim()

    main_agent = build_agent(MAIN_AGENT_NAME, instrument, contingent_claim)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True,
    )

    optimizer_cls = tf.keras.optimizers.Adam if main_agent.is_trainable else None
    learning_rate = learning_rate_schedule if main_agent.is_trainable else None

    env = Environment(
        agent=main_agent,
        T=T,
        N=N,
        r=R,
        instrument_list=instruments,
        n_instruments=1,
        contingent_claim=contingent_claim,
        cost_function=ProportionalCost(PROPORTIONAL_COST),
        risk_measure=CVaR(alpha=CVaR_ALPHA),
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        optimizer=optimizer_cls,
        resample_each_epoch=RESAMPLE_EACH_EPOCH,
        train_random_seed=GLOBAL_RANDOM_SEED,
    )

    maybe_load_trainable_agent(main_agent, MAIN_AGENT_NAME, env)

    if RUN_MODE in {"train", "train_and_evaluate"}:
        if not main_agent.is_trainable:
            raise ValueError(f"Main agent '{MAIN_AGENT_NAME}' is not trainable. Choose a trainable agent for train mode.")
        print("[run] training...")
        env.train(train_paths=TRAIN_PATHS, val_paths=VAL_PATHS, random_seed=GLOBAL_RANDOM_SEED)
        maybe_save_trainable_agent(main_agent, MAIN_AGENT_NAME, env)

    if RUN_MODE in {"evaluate", "train_and_evaluate"}:
        print("[run] evaluating...")
        evaluation_agents = [main_agent]
        for agent_name in COMPARE_AGENT_NAMES:
            compare_agent = build_agent(agent_name, instrument, contingent_claim)
            if compare_agent.is_trainable:
                model_path, _ = get_agent_paths(compare_agent, agent_name)
                if os.path.exists(model_path):
                    compare_agent.load_model(model_path)
                    print(f"[load] comparison model: {model_path}")
                else:
                    print(f"[warn] model not found for {agent_name}: {model_path}")
            evaluation_agents.append(compare_agent)

        os.makedirs(os.path.join(ROOT_DIR, PLOTS_DIR), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, STATS_DIR), exist_ok=True)

        measures = [CVaR(0.5), CVaR(0.95), CVaR(0.99), MAE(), WorstCase()]
        plot_path = os.path.join(
            ROOT_DIR,
            PLOTS_DIR,
            f"{main_agent.name}_{get_model_name_for_agent(MAIN_AGENT_NAME)}_comparison.pdf",
        )
        stats_path = os.path.join(
            ROOT_DIR,
            STATS_DIR,
            f"{main_agent.name}_{get_model_name_for_agent(MAIN_AGENT_NAME)}_comparison.xlsx",
        )

        result = env.terminal_hedging_error_multiple_agents(
            agents=evaluation_agents,
            n_paths=EVAL_PATHS,
            random_seed=EVAL_SEED,
            plot_error=True,
            loss_functions=measures,
            plot_title="Terminal Hedging Error",
            save_plot_path=plot_path,
            save_stats_path=stats_path,
            min_x=PLOT_MIN_X,
            max_x=PLOT_MAX_X,
            language=LANGUAGE,
            pricing_method=PRICING_METHOD,
        )
        print(result)
        print(f"[save] plot: {plot_path}")
        print(f"[save] stats: {stats_path}")


if __name__ == "__main__":
    main()
