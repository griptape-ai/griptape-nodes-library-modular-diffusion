"""Maps a diffusers pipeline class name to a memory-estimation "family" and normalizes
that pipeline's denoiser config into the generic fields the activation formulas need.

Field names for layers/heads/hidden-dim/patch-size are NOT uniform across diffusers
model configs (verified by reading the real constructor signatures of
FluxTransformer2DModel, LTXVideoTransformer3DModel, UNet2DConditionModel,
QwenImageTransformer2DModel, WanTransformer3DModel, SD3Transformer2DModel,
Flux2Transformer2DModel, ZImageTransformer2DModel, HunyuanVideo15Transformer3DModel,
LTX2VideoTransformer3DModel, MiniMaxH3Transformer3DModel) -- each pipeline class gets
its own small adapter rather than one generic `pipe.transformer.config.X` lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]


class MemoryFamily(Enum):
    """Architecture family used to pick a denoiser activation-memory formula."""

    UNET_SDPA = "unet_sdpa"
    IMAGE_DIT_SDPA = "image_dit_sdpa"
    VIDEO_DIT_JOINT_SDPA = "video_dit_joint_sdpa"


# Keyed by the same pipeline-class-name strings as
# latent_pipeline_drivers/driver_factory.py's _DRIVER_REGISTRY (one entry per key there;
# a unit test asserts the two registries' key sets stay in sync).
_FAMILY_BY_PIPELINE_NAME: dict[str, MemoryFamily] = {
    "FluxFillPipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "HunyuanVideo15Pipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "HunyuanVideo15ImageToVideoPipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "FluxKontextPipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "FluxPipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "Flux2Pipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "Flux2KleinPipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "LTX2Pipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "MiniMaxH3ModularPipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "QwenImagePipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "QwenImageEditPipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "StableDiffusion3Pipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "StableDiffusionXLPipeline": MemoryFamily.UNET_SDPA,
    "ZImagePipeline": MemoryFamily.IMAGE_DIT_SDPA,
    "WanPipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "LTXPipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "WanImageToVideoPipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "WanAnimatePipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
    "WanVACEPipeline": MemoryFamily.VIDEO_DIT_JOINT_SDPA,
}


def get_memory_family(pipeline_name: str) -> MemoryFamily | None:
    """Return the memory family for *pipeline_name*, or None if unregistered."""
    return _FAMILY_BY_PIPELINE_NAME.get(pipeline_name)


@dataclass(frozen=True)
class DenoiserConfigFields:
    """Generic denoiser shape used by the Image-DiT-SDPA and Video-DiT-joint-SDPA formulas."""

    hidden_dim: int
    num_layers: int
    patch_size_spatial: int
    patch_size_temporal: int


@dataclass(frozen=True)
class UnetLevelConfig:
    """Per-resolution-level shape used by the UNet-SDPA formula (SDXL only)."""

    hidden_dim: int
    num_transformer_layers: int
    downsample_factor: int


def _adapt_flux_family(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    return DenoiserConfigFields(
        hidden_dim=cfg.num_attention_heads * cfg.attention_head_dim,
        num_layers=cfg.num_layers + cfg.num_single_layers,
        patch_size_spatial=cfg.patch_size,
        patch_size_temporal=1,
    )


def _adapt_qwen_family(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    return DenoiserConfigFields(
        hidden_dim=cfg.num_attention_heads * cfg.attention_head_dim,
        num_layers=cfg.num_layers,
        patch_size_spatial=cfg.patch_size,
        patch_size_temporal=1,
    )


def _adapt_sd3(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    return DenoiserConfigFields(
        hidden_dim=cfg.num_attention_heads * cfg.attention_head_dim,
        num_layers=cfg.num_layers,
        patch_size_spatial=cfg.patch_size,
        patch_size_temporal=1,
    )


def _adapt_zimage(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    return DenoiserConfigFields(
        hidden_dim=cfg.dim,
        num_layers=cfg.n_layers + cfg.n_refiner_layers,
        patch_size_spatial=cfg.all_patch_size[0],
        patch_size_temporal=1,
    )


def _adapt_ltx_family(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    return DenoiserConfigFields(
        hidden_dim=cfg.num_attention_heads * cfg.attention_head_dim,
        num_layers=cfg.num_layers,
        patch_size_spatial=cfg.patch_size,
        patch_size_temporal=cfg.patch_size_t,
    )


def _adapt_wan_family(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    patch_t, patch_h, _patch_w = cfg.patch_size
    return DenoiserConfigFields(
        hidden_dim=cfg.num_attention_heads * cfg.attention_head_dim,
        num_layers=cfg.num_layers,
        patch_size_spatial=patch_h,
        patch_size_temporal=patch_t,
    )


def _adapt_hunyuan_video15(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    return DenoiserConfigFields(
        hidden_dim=cfg.num_attention_heads * cfg.attention_head_dim,
        num_layers=cfg.num_layers + cfg.num_refiner_layers,
        patch_size_spatial=cfg.patch_size,
        patch_size_temporal=cfg.patch_size_t,
    )


def _adapt_minimax_h3(pipe: DiffusionPipeline) -> DenoiserConfigFields:
    cfg = pipe.transformer.config
    patch_t, patch_h, _patch_w = cfg.patch_size
    return DenoiserConfigFields(
        hidden_dim=cfg.hidden_size,
        num_layers=cfg.num_layers + cfg.num_refiner_layers,
        patch_size_spatial=patch_h,
        patch_size_temporal=patch_t,
    )


# Reused across sibling pipeline classes that share the same transformer class
# (verified for Flux/Flux2 family and LTX/LTX2; the remaining reuses -- Flux Fill/
# Kontext, Qwen Edit, WAN i2v/VACE/Animate, HunyuanVideo15 i2v -- are assumed to inherit
# their base family's transformer config via driver naming, per the spike's own
# unresolved caveat; spot-check before shipping.)
_DENOISER_FIELD_ADAPTERS: dict[str, Callable[[DiffusionPipeline], DenoiserConfigFields]] = {
    "FluxPipeline": _adapt_flux_family,
    "FluxFillPipeline": _adapt_flux_family,
    "FluxKontextPipeline": _adapt_flux_family,
    "Flux2Pipeline": _adapt_flux_family,
    "Flux2KleinPipeline": _adapt_flux_family,
    "QwenImagePipeline": _adapt_qwen_family,
    "QwenImageEditPipeline": _adapt_qwen_family,
    "StableDiffusion3Pipeline": _adapt_sd3,
    "ZImagePipeline": _adapt_zimage,
    "LTXPipeline": _adapt_ltx_family,
    "LTX2Pipeline": _adapt_ltx_family,
    "WanPipeline": _adapt_wan_family,
    "WanImageToVideoPipeline": _adapt_wan_family,
    "WanVACEPipeline": _adapt_wan_family,
    "WanAnimatePipeline": _adapt_wan_family,
    "HunyuanVideo15Pipeline": _adapt_hunyuan_video15,
    "HunyuanVideo15ImageToVideoPipeline": _adapt_hunyuan_video15,
    "MiniMaxH3ModularPipeline": _adapt_minimax_h3,
}


def get_denoiser_config_fields(pipeline_name: str, pipe: DiffusionPipeline) -> DenoiserConfigFields | None:
    """Return the normalized denoiser config for *pipeline_name*, or None if unregistered."""
    adapter = _DENOISER_FIELD_ADAPTERS.get(pipeline_name)
    if adapter is None:
        return None
    return adapter(pipe)


def get_sdxl_unet_levels(pipe: DiffusionPipeline) -> list[UnetLevelConfig]:
    """Read SDXL's UNet as a list of per-resolution-level shapes.

    UNet2DConditionModel has no flat num_layers/hidden_dim -- block_out_channels gives
    the per-level channel width, and transformer_layers_per_block (or layers_per_block
    as a fallback) gives the per-level transformer-layer count. Downsampling follows
    the standard stride-2-per-level convention.
    """
    cfg = pipe.unet.config
    block_out_channels = cfg.block_out_channels
    transformer_layers_per_block = cfg.transformer_layers_per_block
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(block_out_channels)

    levels = []
    for level_index, (channels, num_transformer_layers) in enumerate(
        zip(block_out_channels, transformer_layers_per_block, strict=True)
    ):
        levels.append(
            UnetLevelConfig(
                hidden_dim=channels,
                num_transformer_layers=num_transformer_layers,
                downsample_factor=2**level_index,
            )
        )
    return levels
