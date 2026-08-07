"""Two-candidate recursive observer for belief-weighted guidance."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class ObserverConfig:
    """Residual-scoring and guidance settings for the online observer."""

    temperature: float
    num_samples: int = 8
    tau_min: float = 0.3
    tau_max: float = 0.7
    action_dims: int = 8
    belief_weighted: bool = False
    guidance_lambda: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if not 0 <= self.tau_min < self.tau_max <= 1:
            raise ValueError(f"Expected 0 <= tau_min < tau_max <= 1, got [{self.tau_min}, {self.tau_max}]")
        if self.action_dims <= 0:
            raise ValueError(f"action_dims must be positive, got {self.action_dims}")
        if self.guidance_lambda < 0:
            raise ValueError(f"guidance_lambda must be non-negative, got {self.guidance_lambda}")


class BeliefFilter:
    """Uniform-prior, log-space Bayes filter over [positive, negative]."""

    def __init__(self, temperature: float):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self._temperature = float(temperature)
        self.reset()

    def reset(self) -> None:
        self._log_belief = np.full(2, -np.log(2.0), dtype=np.float64)
        self._num_updates = 0

    def update(self, energies: np.ndarray) -> np.ndarray:
        energies = np.asarray(energies, dtype=np.float64)
        if energies.shape != (2,):
            raise ValueError(f"Expected [positive, negative] energies, got shape {energies.shape}")
        if not np.all(np.isfinite(energies)):
            raise ValueError(f"Energies must be finite, got {energies}")

        self._log_belief -= energies / self._temperature
        maximum = np.max(self._log_belief)
        self._log_belief -= maximum + np.log(np.exp(self._log_belief - maximum).sum())
        self._num_updates += 1
        return self.belief

    @property
    def belief(self) -> np.ndarray:
        return np.exp(self._log_belief)

    @property
    def positive(self) -> float:
        return float(self.belief[0])

    @property
    def negative(self) -> float:
        return float(self.belief[1])

    @property
    def num_updates(self) -> int:
        return self._num_updates
