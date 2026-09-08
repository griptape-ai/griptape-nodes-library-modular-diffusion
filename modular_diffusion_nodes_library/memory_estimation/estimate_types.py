"""Return types for pipeline memory estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_BYTES_PER_GB = 1024**3


def _bytes_to_gb(num_bytes: int) -> float:
    """Convert a byte count to GB, rounded to a sane display precision.

    `to_dict()` is the API's output boundary (see docs/spikes/
    memory_estimation_preload_api_plan.md, Decision 6) -- raw byte counts aren't a
    useful unit for a caller to read, so it reports GB floats instead. Internal fields
    stay byte-precise ints since topology math (sum/max, quantization multipliers)
    needs that precision.
    """
    return round(num_bytes / _BYTES_PER_GB, 4)


@dataclass(frozen=True)
class ComponentMemoryEstimate:
    """Memory estimate for a single resident pipeline component (e.g. a transformer or a VAE)."""

    component_name: str
    role: str
    weight_bytes: int
    activation_bytes: int
    total_bytes: int
    is_estimated: bool = False
    warning: str | None = None

    @classmethod
    def create(
        cls,
        component_name: str,
        role: str,
        weight_bytes: int,
        activation_bytes: int,
        *,
        is_estimated: bool = False,
        warning: str | None = None,
    ) -> ComponentMemoryEstimate:
        return cls(
            component_name=component_name,
            role=role,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            total_bytes=weight_bytes + activation_bytes,
            is_estimated=is_estimated,
            warning=warning,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "role": self.role,
            "weight_gb": _bytes_to_gb(self.weight_bytes),
            "activation_gb": _bytes_to_gb(self.activation_bytes),
            "total_gb": _bytes_to_gb(self.total_bytes),
            "is_estimated": self.is_estimated,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class PipelineMemoryEstimate:
    """Per-component memory estimate for an entire pipeline, loaded or not.

    `basis` records which estimator produced this: "loaded" (exact weights, walked off
    a resident pipeline) or "config_only" (pre-load, meta-device-derived weights).
    `confidence` is a coarse qualitative label, not a calibrated percentage -- see
    docs/spikes/memory_estimation_preload_api_plan.md, Decision 6. It is "low" exactly
    when the config-only estimate had to assume a conservative offload topology because
    `memory_optimization_strategy` is "Automatic" (Decision 4); "high" otherwise.
    """

    pipeline_name: str
    offload_mode: str | None
    components: list[ComponentMemoryEstimate]
    estimated_peak_bytes: int
    basis: str = "loaded"
    confidence: str = "high"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "offload_mode": self.offload_mode,
            "components": [component.to_dict() for component in self.components],
            "estimated_peak_gb": _bytes_to_gb(self.estimated_peak_bytes),
            "basis": self.basis,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }
