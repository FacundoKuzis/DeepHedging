from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np


@dataclass
class ContinuationContext:
    claim: Any
    instrument: Any
    n_steps: int
    maturity: float
    dt: float
    strike: float | None = None
    option_type: str | None = None
    random_seed: int | None = None


class ContinuationValueProvider(ABC):
    """Interface for continuation value estimators C_{t+1}."""

    provider_name = "base"

    def __init__(self) -> None:
        self.context: ContinuationContext | None = None

    def prepare(self, context: ContinuationContext) -> None:
        self.context = context

    def supports_claim(self, claim: Any) -> bool:
        _ = claim
        return True

    @abstractmethod
    def estimate_continuation_t1(
        self,
        spot_t1: np.ndarray,
        t_index: int,
        path_prefix: np.ndarray | None = None,
        per_path_r: np.ndarray | float | None = None,
        per_path_sigma: np.ndarray | float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Return continuation values at t+1.

        Args:
            spot_t1: array (batch, n_outer)
            t_index: hedge index t in [0, N-1]
            path_prefix: optional observed path up to t, shape (batch, t+1)
            per_path_r/per_path_sigma: scalar or array(batch,)

        Returns:
            continuation values with shape (batch, n_outer)
        """

    def get_price_batch(
        self,
        path_s0: np.ndarray,
        path_r: np.ndarray | float,
        path_sigma: np.ndarray | float,
    ) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_price_batch."
        )


def _resolve_provider_name(name: str) -> str:
    value = str(name).strip().lower()
    aliases = {
        "bs": "bs_closed_form",
        "black_scholes": "bs_closed_form",
        "blackscholes": "bs_closed_form",
        "mc": "monte_carlo",
        "lsmc": "lsm",
        "asian_mc": "asian_monte_carlo",
        "asian_lsm": "asian_lsmc",
    }
    return aliases.get(value, value)


def build_continuation_provider(provider_name: str, **kwargs: Any) -> ContinuationValueProvider:
    normalized = _resolve_provider_name(provider_name)
    # Local import to avoid circular imports.
    from DeepHedging.utils.lrm_providers import (  # pylint: disable=import-outside-toplevel
        AsianLSMContinuationProvider,
        AsianMonteCarloContinuationProvider,
        BSClosedFormContinuationProvider,
        LSMContinuationProvider,
        MonteCarloContinuationProvider,
    )

    mapping = {
        "bs_closed_form": BSClosedFormContinuationProvider,
        "monte_carlo": MonteCarloContinuationProvider,
        "lsm": LSMContinuationProvider,
        "asian_monte_carlo": AsianMonteCarloContinuationProvider,
        "asian_lsmc": AsianLSMContinuationProvider,
    }
    if normalized not in mapping:
        raise ValueError(
            "benchmark_lrm_provider must be one of "
            "{'bs_closed_form','monte_carlo','lsm','asian_monte_carlo','asian_lsmc'}."
        )
    provider_cls = mapping[normalized]
    init_params = inspect.signature(provider_cls.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
    return provider_cls(**filtered_kwargs)
