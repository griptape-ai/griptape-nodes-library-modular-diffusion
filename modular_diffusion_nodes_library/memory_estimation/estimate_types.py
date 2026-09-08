"""Return types for pipeline memory estimation."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class PipelineMemoryEstimate:
    """Per-component memory estimate for an entire loaded pipeline."""

    pipeline_name: str
    offload_mode: str | None
    components: list[ComponentMemoryEstimate]
    estimated_peak_bytes: int
    warnings: list[str] = field(default_factory=list)
