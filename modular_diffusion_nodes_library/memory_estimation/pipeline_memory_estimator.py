"""Orchestrates a per-component memory estimate for an already-loaded diffusion pipeline.

This is the only module in `memory_estimation` that touches the real `pipe` object.
Everything it calls (activation_formulas, vae_formula, text_encoder_formula,
family_registry) is pure/testable with synthetic inputs.
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact, ComponentSourceType
from modular_diffusion_nodes_library.component_loading.component_slots import ALLOWED_COMPONENT_SLOTS
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class
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
from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import (
    ComponentConfigNotCachedError,
    build_component_on_meta_device,
    resolve_base_component_config,
)
from modular_diffusion_nodes_library.memory_estimation.quantization_bytes import resolve_effective_bytes_per_element
from modular_diffusion_nodes_library.memory_estimation.text_encoder_formula import (
    estimate_text_encoder_activation_bytes,
)
from modular_diffusion_nodes_library.memory_estimation.vae_formula import estimate_vae_activation_bytes
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.pipeline_utils import MEMORY_HEADROOM_FACTOR, detect_offload_method
from modular_diffusion_nodes_library.utils.torch_utils import get_model_memory

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
    from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact

logger = logging.getLogger("modular_diffusers_nodes_library")

# Base-pipeline components are loaded at a hardcoded dtype today (e.g.
# flux_parameters.py:65) -- no per-pipeline dtype selection exists yet to read from the
# artifact, so the pre-load estimator assumes the same hardcoded value.
_BASE_PIPELINE_TORCH_DTYPE = torch.bfloat16

_WEIGHT_BEARING_ROLES = {"text_encoder", "denoiser", "vae"}


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


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.zeros(1, dtype=dtype).element_size()


def _resolve_pipeline_cls(artifact: DiffusionPipelineArtifact) -> type:
    if not artifact.builder_module or not artifact.builder_class_name:
        msg = (
            f"Attempted to estimate memory for pipeline '{artifact.pipeline_name}'. "
            f"Failed because the artifact has no builder_module/builder_class_name."
        )
        raise ValueError(msg)
    builder_class = getattr(import_module(artifact.builder_module), artifact.builder_class_name)
    return builder_class.pipeline_cls()


def _weight_bearing_slots(
    pipeline_cls: type, component_overrides: dict[str, ComponentArtifact], *, all_overrides: bool
) -> list[str]:
    if all_overrides:
        return [slot for slot in component_overrides if _classify_role(slot) in _WEIGHT_BEARING_ROLES]
    all_slots, _ = pipeline_cls._get_signature_keys(pipeline_cls)  # noqa: SLF001
    return [
        slot for slot in ALLOWED_COMPONENT_SLOTS if slot in all_slots and _classify_role(slot) in _WEIGHT_BEARING_ROLES
    ]


def _stored_bytes_for_override(
    override: ComponentArtifact, effective_slot: str, pipeline_cls: type
) -> tuple[int, int, torch.nn.Module | None]:
    """Return (weight_bytes, stored_bytes_per_element, meta_component_or_None) for an
    override component, at its own declared dtype -- before any dynamic-quantization
    multiplier.

    `effective_slot` is the pipeline slot this override fills (the `_component_
    overrides` dict key), which takes precedence over `override.component` for class
    lookup -- mirrors `ComponentArtifact.materialize()` (component_artifact.py:106-119),
    since a generic override can be wired to a differently-named slot.

    Config is always attempted (best-effort) so activation formulas have shape data to
    read even for a SINGLE_FILE source. Weight bytes for a SINGLE_FILE source come from
    the file's on-disk size directly (Decision 5: it's one unsharded file, and for a
    GGUF file that size already reflects its final quantized bytes) rather than the
    meta-built component's shape-derived count.
    """
    torch_dtype = getattr(torch, override.torch_dtype)
    config_dict = override.try_read_config(pipeline_cls=pipeline_cls)
    meta_component = None
    if config_dict is not None:
        component_cls = get_component_class(pipeline_cls, effective_slot)
        meta_component = build_component_on_meta_device(component_cls, config_dict, torch_dtype)

    if override.source_type == ComponentSourceType.SINGLE_FILE:
        if not override.file_path:
            msg = (
                f"Attempted to estimate memory for override component '{override.component}'. "
                f"Failed because it has no file_path."
            )
            raise ValueError(msg)
        weight_bytes = Path(override.file_path).stat().st_size
        return weight_bytes, _dtype_element_size(torch_dtype), meta_component

    if meta_component is None:
        msg = (
            f"Attempted to estimate memory for override component '{override.component}'. "
            f"Failed because its config is not in the local HuggingFace cache."
        )
        raise ComponentConfigNotCachedError(msg)
    return get_model_memory(meta_component), _element_size(meta_component), meta_component


def _stored_bytes_for_base_component(
    slot: str,
    pipeline_cls: type,
    base_repo_id: str,
    base_revision: str | None,
    torch_dtype: torch.dtype,
) -> tuple[int, int, torch.nn.Module]:
    config_dict = resolve_base_component_config(base_repo_id, slot, base_revision)
    component_cls = get_component_class(pipeline_cls, slot)
    meta_component = build_component_on_meta_device(component_cls, config_dict, torch_dtype)
    return get_model_memory(meta_component), _element_size(meta_component), meta_component


def estimate_pipeline_memory_from_build_data(
    artifact: DiffusionPipelineArtifact,
    latent: LatentArtifact,
) -> PipelineMemoryEstimate:
    """Estimate memory for a pipeline that has not been built yet.

    Never touches weight files or triggers a download: component configs are read from
    the warm HuggingFace cache and each component is built on the meta device
    (meta_device_builder.py) to get exact parameter shapes without allocating storage.
    See docs/spikes/memory_estimation_preload_api_plan.md.
    """
    build_data = artifact.build_data
    optimization_kwargs = artifact.optimization_kwargs
    pipeline_name = artifact.pipeline_name
    pipeline_cls = _resolve_pipeline_cls(artifact)

    component_overrides: dict[str, ComponentArtifact] = build_data.get("_component_overrides", {})
    is_all_overrides = bool(build_data.get("_all_overrides"))
    base_repo_id = build_data.get("base_repo_id")
    base_revision = build_data.get("base_revision")

    if not is_all_overrides and not base_repo_id:
        msg = (
            f"Attempted to estimate memory for pipeline '{pipeline_name}'. "
            f"Failed because build_data has no 'base_repo_id' and is not built purely from overrides."
        )
        raise ValueError(msg)

    slots = _weight_bearing_slots(pipeline_cls, component_overrides, all_overrides=is_all_overrides)
    quantization_mode = optimization_kwargs.get("quantization_mode", "None")
    transformer_layerwise_casting = bool(optimization_kwargs.get("transformer_layerwise_casting", False))
    memory_optimization_strategy = optimization_kwargs.get("memory_optimization_strategy", "Manual")

    components: list[ComponentMemoryEstimate] = []
    denoiser_num_layers: int | None = None

    for slot in slots:
        role = _classify_role(slot)
        override = component_overrides.get(slot)
        try:
            if override is not None:
                weight_bytes, stored_bytes_per_element, meta_component = _stored_bytes_for_override(
                    override, slot, pipeline_cls
                )
            else:
                weight_bytes, stored_bytes_per_element, meta_component = _stored_bytes_for_base_component(
                    slot, pipeline_cls, base_repo_id, base_revision, _BASE_PIPELINE_TORCH_DTYPE
                )
        except ComponentConfigNotCachedError as e:
            components.append(ComponentMemoryEstimate.create(slot, role, 0, 0, is_estimated=True, warning=str(e)))
            continue

        effective_bytes_per_element = resolve_effective_bytes_per_element(
            stored_bytes_per_element,
            slot=slot,
            quantization_mode=quantization_mode,
            transformer_layerwise_casting=transformer_layerwise_casting,
            is_prequantized=artifact.is_prequantized,
            supports_layerwise_casting=artifact.supports_layerwise_casting,
        )
        weight_bytes = int(weight_bytes / stored_bytes_per_element * effective_bytes_per_element)
        element_size = int(effective_bytes_per_element)

        if role == "text_encoder":
            activation_bytes = estimate_text_encoder_activation_bytes()
            components.append(ComponentMemoryEstimate.create(slot, role, weight_bytes, activation_bytes))
            continue

        if role == "vae":
            if meta_component is None:
                warning = "VAE config not available; showing weights only."
                components.append(
                    ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
                )
                continue
            if optimization_kwargs.get("vae_tiling") and hasattr(meta_component, "enable_tiling"):
                meta_component.enable_tiling()
            try:
                activation_bytes = estimate_vae_activation_bytes(
                    meta_component, latent, optimization_kwargs, element_size
                )
            except (AttributeError, KeyError, TypeError, IndexError) as e:
                warning = f"Failed to estimate VAE activation memory: {e}. Showing weights only."
                components.append(
                    ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
                )
                continue
            components.append(ComponentMemoryEstimate.create(slot, role, weight_bytes, activation_bytes))
            continue

        # role == "denoiser"
        if meta_component is None:
            warning = "Denoiser config not available; showing weights only."
            components.append(
                ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
            )
            continue

        denoiser_pipe_stub = SimpleNamespace(**{slot: meta_component})
        activation_bytes, warning = _estimate_denoiser_activation_bytes(
            meta_component, denoiser_pipe_stub, pipeline_name, latent, element_size
        )
        family = get_memory_family(pipeline_name)
        if family is not None and family != MemoryFamily.UNET_SDPA:
            try:
                fields = get_denoiser_config_fields(pipeline_name, denoiser_pipe_stub)
            except (AttributeError, KeyError, TypeError, IndexError):
                fields = None
            if fields is not None:
                denoiser_num_layers = fields.num_layers
        components.append(
            ComponentMemoryEstimate.create(
                slot, role, weight_bytes, activation_bytes, is_estimated=warning is not None, warning=warning
            )
        )

    if memory_optimization_strategy == "Automatic":
        # Conservative upper bound: assume no offload, since the real Automatic
        # cascade depends on free VRAM at build time and can't be predicted here.
        # See docs/spikes/memory_estimation_preload_api_plan.md, Decision 4.
        offload_mode = None
        confidence = "low"
        automatic_warning = (
            "memory_optimization_strategy is 'Automatic'; the actual offload topology depends on free VRAM "
            "at build time and cannot be predicted before loading. Showing the upper bound (no offload "
            "assumed) -- actual usage after building will be equal to or lower than this. Set the strategy "
            "to Manual for an exact pre-load estimate."
        )
    else:
        # Translate the UI's cpu_offload_strategy vocabulary ("None"/"Model"/
        # "Sequential") into the same lowercase vocabulary detect_offload_method()
        # returns for the post-load path ("model"/"sequential"/None), so offload_mode
        # means the same thing regardless of which estimator produced it.
        cpu_offload_strategy = optimization_kwargs.get("cpu_offload_strategy", "None")
        offload_mode = {"None": None, "Model": "model", "Sequential": "sequential"}.get(cpu_offload_strategy)
        confidence = "high"
        automatic_warning = None

    peak_weight_bytes = _compute_peak_weight_topology(components, offload_mode, denoiser_num_layers)
    total_activation_bytes = sum(c.activation_bytes for c in components)
    estimated_peak_bytes = int((peak_weight_bytes + total_activation_bytes) * MEMORY_HEADROOM_FACTOR)

    warnings = [c.warning for c in components if c.warning is not None]
    if automatic_warning is not None:
        warnings.append(automatic_warning)

    return PipelineMemoryEstimate(
        pipeline_name=pipeline_name,
        offload_mode=offload_mode,
        components=components,
        estimated_peak_bytes=estimated_peak_bytes,
        basis="config_only",
        confidence=confidence,
        warnings=warnings,
    )


def estimate_pipeline_memory_from_artifact(
    artifact: DiffusionPipelineArtifact,
    latent: LatentArtifact,
) -> PipelineMemoryEstimate:
    """Single public entry point: estimate memory for `artifact`, regardless of whether
    its pipeline has been built yet.

    Dispatches to the exact post-load estimator when the pipeline is already resident
    in model_cache under artifact.config_hash, otherwise to the pre-load, config-only
    estimator. See docs/spikes/memory_estimation_preload_api_plan.md, Decision 1.
    """
    if artifact.config_hash and model_cache.has_pipeline(artifact.config_hash):
        pipe = model_cache.get_pipeline(artifact.config_hash)
        return estimate_pipeline_memory(pipe, latent, artifact.optimization_kwargs, artifact.pipeline_name)
    return estimate_pipeline_memory_from_build_data(artifact, latent)
