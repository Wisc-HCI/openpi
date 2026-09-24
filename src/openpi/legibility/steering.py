"""Validated, request-owned steering settings (wire protocol version 1)."""

from collections.abc import Mapping
import dataclasses
import math
import numbers

PROTOCOL_VERSION = 1
MODES = ("off", "fixed", "time_decay", "belief")


@dataclasses.dataclass(frozen=True)
class SteeringConfig:
    mode: str = "off"
    initial_scale: float = 1.0
    decay: float = 0.9
    belief_temperature: float | None = None
    belief_lambda: float = 1.0
    belief_samples: int = 8
    belief_tau_min: float = 0.3
    belief_tau_max: float = 0.7
    belief_action_dims: int = 8

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in MODES:
            raise ValueError(f"steering.mode must be one of {MODES}")
        for name in ("initial_scale", "decay", "belief_lambda", "belief_tau_min", "belief_tau_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"steering.{name} must be a finite number")
        if self.initial_scale < 0 or self.belief_lambda < 0:
            raise ValueError("steering initial_scale and belief_lambda must be non-negative")
        if not 0 <= self.decay <= 1:
            raise ValueError("steering.decay must be in [0, 1]")
        for name in ("belief_samples", "belief_action_dims"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"steering.{name} must be a positive integer")
        if not 0 <= self.belief_tau_min < self.belief_tau_max <= 1:
            raise ValueError("steering belief tau range must satisfy 0 <= min < max <= 1")
        if self.mode == "belief":
            value = self.belief_temperature
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
                raise ValueError("belief mode requires a finite positive belief_temperature")
        elif self.belief_temperature is not None:
            raise ValueError("belief_temperature applies only to steering mode 'belief'")

    @classmethod
    def from_request(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("steering must be a configuration dictionary")
        unknown = set(value) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown steering fields: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self):
        return dataclasses.asdict(self)
