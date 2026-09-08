"""Orchestrates a per-component memory estimate for an already-loaded diffusion pipeline.

This is the only module in `memory_estimation` that touches the real `pipe` object.
Everything it calls (activation_formulas, vae_formula, text_encoder_formula,
family_registry) is pure/testable with synthetic inputs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from modular_diffusion_nodes_library.memory_estimation.activation_formulas import (
    estimate_image_dit_sdpa_activation_bytes,
    estimate_unet_sdpa_activation_bytes,
    estimate_video_dit_joint_sdpa_activation_bytes,
)
from modular_diffusion_nodes_library.memory_estimation.estimate_types import (
    ComponentMemoryEstimate,
    PipelineMemoryEstimate,
)
from modular_diffusion_nodes_library.memory_estimation.family_registry import (
    MemoryFamily,
    get_denoiser_config_fields,
    get_memory_family,
    get_sdxl_unet_levels,
)
from modular_diffusion_nodes_library.memory_estimation.text_encoder_formula import (
    estimate_text_encoder_activation_bytes,
)
from modular_diffusion_nodes_library.memory_estimation.vae_formula import estimate_vae_activation_bytes
from modular_diffusion_nodes_library.utils.pipeline_utils import MEMORY_HEADROOM_FACTOR, detect_offload_method
from modular_diffusion_nodes_library.utils.torch_utils import get_model_memory

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact

logger = logging.getLogger("modular_diffusers_nodes_library")


def _classify_role(component_name: str) -> str:
    if component_name.startswith("text_encoder"):
        return "text_encoder"
    if component_name in ("transformer", "unet"):
        return "denoiser"
    if component_name == "vae":
        return "vae"
    return "other"


def _element_size(component: torch.nn.Module) -> int:
    try:
        return next(component.parameters()).element_size()
    except StopIteration:
        return 4  # no parameters found; fall back to float32 width


def _estimate_denoiser_activation_bytes(
    component: torch.nn.Module,
    pipe: DiffusionPipeline,
    pipeline_name: str,
    latent: LatentArtifact,
    element_size: int,
) -> tuple[int, str | None]:
    family = get_memory_family(pipeline_name)
    if family is None:
        return 0, f"No memory-estimation family registered for pipeline '{pipeline_name}'; showing weights only."

    latent_shape = latent.shape
    height_latent, width_latent = latent_shape[-2], latent_shape[-1]

    try:
        if family == MemoryFamily.UNET_SDPA:
            levels = get_sdxl_unet_levels(pipe)
            activation_bytes = estimate_unet_sdpa_activation_bytes(levels, height_latent, width_latent, element_size)
            return activation_bytes, None

        fields = get_denoiser_config_fields(pipeline_name, pipe)
        if fields is None:
            return 0, f"No denoiser config adapter registered for pipeline '{pipeline_name}'; showing weights only."

        if family == MemoryFamily.IMAGE_DIT_SDPA:
            activation_bytes = estimate_image_dit_sdpa_activation_bytes(
                height_latent,
                width_latent,
                fields.hidden_dim,
                fields.num_layers,
                fields.patch_size_spatial,
                element_size,
            )
            return activation_bytes, None

        num_frames_latent = latent_shape[-3] if len(latent_shape) >= 5 else 1  # noqa: PLR2004
        activation_bytes = estimate_video_dit_joint_sdpa_activation_bytes(
            num_frames_latent,
            height_latent,
            width_latent,
            fields.hidden_dim,
            fields.num_layers,
            fields.patch_size_spatial,
            fields.patch_size_temporal,
            element_size,
        )
        return activation_bytes, None
    except (AttributeError, KeyError, TypeError, IndexError) as e:
        return 0, f"Failed to read denoiser config for pipeline '{pipeline_name}': {e}. Showing weights only."


def _estimate_component(
    component_name: str,
    component: torch.nn.Module,
    role: str,
    pipe: DiffusionPipeline,
    pipeline_name: str,
    latent: LatentArtifact,
    optimization_kwargs: dict[str, Any],
) -> ComponentMemoryEstimate:
    weight_bytes = get_model_memory(component)
    element_size = _element_size(component)

    if role == "text_encoder":
        activation_bytes = estimate_text_encoder_activation_bytes()
        return ComponentMemoryEstimate.create(component_name, role, weight_bytes, activation_bytes)

    if role == "vae":
        try:
            activation_bytes = estimate_vae_activation_bytes(component, latent, optimization_kwargs, element_size)
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            warning = f"Failed to estimate VAE activation memory: {e}. Showing weights only."
            return ComponentMemoryEstimate.create(
                component_name, role, weight_bytes, 0, is_estimated=True, warning=warning
            )
        return ComponentMemoryEstimate.create(component_name, role, weight_bytes, activation_bytes)

    if role == "denoiser":
        activation_bytes, warning = _estimate_denoiser_activation_bytes(
            component, pipe, pipeline_name, latent, element_size
        )
        return ComponentMemoryEstimate.create(
            component_name, role, weight_bytes, activation_bytes, is_estimated=warning is not None, warning=warning
        )

    warning = f"Unrecognized component '{component_name}'; only weight memory could be estimated."
    return ComponentMemoryEstimate.create(component_name, role, weight_bytes, 0, is_estimated=True, warning=warning)


def _compute_peak_weight_topology(
    components: list[ComponentMemoryEstimate],
    offload_mode: str | None,
    denoiser_num_layers: int | None,
) -> int:
    if not components:
        return 0
    if offload_mode is None:
        return sum(c.weight_bytes for c in components)
    if offload_mode == "model":
        return max(c.weight_bytes for c in components)
    # "sequential": approximate one denoiser layer's weight size, full weight for the
    # rest. Rough approximation, flagged for revisit if real-world accuracy misses target.
    largest = max(components, key=lambda c: c.weight_bytes)
    if largest.role == "denoiser" and denoiser_num_layers:
        per_layer = largest.weight_bytes // denoiser_num_layers
        others = [c.weight_bytes for c in components if c is not largest]
        return per_layer + max(others, default=0)
    return max(c.weight_bytes for c in components)


def estimate_pipeline_memory(
    pipe: DiffusionPipeline,
    latent: LatentArtifact,
    optimization_kwargs: dict[str, Any],
    pipeline_name: str,
) -> PipelineMemoryEstimate:
    """Estimate per-component memory usage for an already-loaded pipeline. Never executes the pipeline."""
    offload_mode = detect_offload_method(pipe)

    components: list[ComponentMemoryEstimate] = []
    for component_name, component in pipe.components.items():
        if component is None or not hasattr(component, "parameters"):
            continue
        role = _classify_role(component_name)
        components.append(
            _estimate_component(component_name, component, role, pipe, pipeline_name, latent, optimization_kwargs)
        )

    denoiser_num_layers = None
    family = get_memory_family(pipeline_name)
    if family is not None and family != MemoryFamily.UNET_SDPA:
        try:
            fields = get_denoiser_config_fields(pipeline_name, pipe)
        except (AttributeError, KeyError, TypeError, IndexError):
            fields = None
        if fields is not None:
            denoiser_num_layers = fields.num_layers

    peak_weight_bytes = _compute_peak_weight_topology(components, offload_mode, denoiser_num_layers)
    total_activation_bytes = sum(c.activation_bytes for c in components)
    estimated_peak_bytes = int((peak_weight_bytes + total_activation_bytes) * MEMORY_HEADROOM_FACTOR)

    warnings = [c.warning for c in components if c.warning is not None]

    return PipelineMemoryEstimate(
        pipeline_name=pipeline_name,
        offload_mode=offload_mode,
        components=components,
        estimated_peak_bytes=estimated_peak_bytes,
        warnings=warnings,
    )


def estimate_component_memory(
    component: torch.nn.Module,
    component_name: str,
    role: str,
    pipe: DiffusionPipeline,
    pipeline_name: str,
    latent: LatentArtifact,
    optimization_kwargs: dict[str, Any],
) -> ComponentMemoryEstimate:
    """Estimate memory for a single component. Exposed independently of estimate_pipeline_memory
    to satisfy "estimate API for a configured pipeline or component" literally.
    """
    return _estimate_component(component_name, component, role, pipe, pipeline_name, latent, optimization_kwargs)
