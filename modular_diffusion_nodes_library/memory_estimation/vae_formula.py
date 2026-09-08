"""Conv/tile-based VAE activation-memory formula.

Deliberately its own module, not folded into activation_formulas.py: VAE activation
memory is conv/resolution-based, not attention/token-based, and it scales with
pixel-space resolution -- it can swing by multiple GB depending on whether tiling is
enabled, which a fixed constant (as used for text encoders) cannot represent.

Tile-size attribute names differ across VAE classes (verified against diffusers
source):
- AutoencoderKL (image): `tile_sample_min_size` (square).
- AutoencoderKLLTXVideo: `tile_sample_min_width` / `tile_sample_min_height` /
  `tile_sample_min_num_frames`.
- AutoencoderKLWan: `tile_sample_min_width` / `tile_sample_min_height` (no temporal
  tiling).

Per-level channel width is summed across resolution levels rather than pairing the
single deepest channel count with the single largest (pixel-space) resolution -- those
never coexist in a real forward pass; the deepest channel count only exists at the
most-downsampled internal resolution. Mirrors the same per-resolution-level shape
already used for the UNet-SDPA family (activation_formulas.py's
estimate_unet_sdpa_activation_bytes / family_registry.get_sdxl_unet_levels).

The config field naming this list comes from also differs across VAE classes:
AutoencoderKL, AutoencoderKLLTXVideo, AutoencoderKLLTX2, AutoencoderKLHunyuanVideo15,
and AutoencoderKLMiniMaxH3 all expose `block_out_channels` directly (verified against
diffusers source). AutoencoderKLWan instead expresses it as `base_dim * dim_mult[i]`
(autoencoder_kl_wan.py:978-981) -- no `block_out_channels` field exists on it at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact


def _resolve_block_out_channels(vae: torch.nn.Module) -> list[int]:
    """Return this VAE's per-level channel widths, regardless of which config field
    name the class uses to express them.
    """
    block_out_channels = getattr(vae.config, "block_out_channels", None)
    if block_out_channels is not None:
        return list(block_out_channels)

    base_dim = getattr(vae.config, "base_dim", None)
    dim_mult = getattr(vae.config, "dim_mult", None)
    if base_dim is not None and dim_mult is not None:
        return [base_dim * multiplier for multiplier in dim_mult]

    msg = (
        f"Attempted to resolve VAE per-level channel widths. Failed because "
        f"'{type(vae).__name__}' exposes neither 'block_out_channels' nor "
        f"'base_dim'/'dim_mult' in its config."
    )
    raise AttributeError(msg)


def _resolve_tiled_extent(vae: torch.nn.Module, source_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return (frames, height, width) actually processed at once, honoring tiling."""
    *_, height, width = source_shape
    num_frames = source_shape[-3] if len(source_shape) >= 5 else 1  # noqa: PLR2004

    if getattr(vae, "tile_sample_min_size", None) is not None:
        tile = vae.tile_sample_min_size
        return num_frames, min(height, tile), min(width, tile)

    tile_height = getattr(vae, "tile_sample_min_height", None)
    tile_width = getattr(vae, "tile_sample_min_width", None)
    if tile_height is not None and tile_width is not None:
        tile_frames = getattr(vae, "tile_sample_min_num_frames", num_frames)
        return min(num_frames, tile_frames), min(height, tile_height), min(width, tile_width)

    return num_frames, height, width


def estimate_vae_activation_bytes(
    vae: torch.nn.Module,
    latent: LatentArtifact,
    optimization_kwargs: dict[str, Any],
    element_size: int,
) -> int:
    """Estimate VAE encode/decode activation memory, honoring vae_tiling/vae_slicing."""
    vae_tiling = optimization_kwargs.get("vae_tiling", False)
    vae_slicing = optimization_kwargs.get("vae_slicing", False)
    source_shape = latent.source_shape

    if vae_tiling and getattr(vae, "use_tiling", False):
        frames, height, width = _resolve_tiled_extent(vae, source_shape)
    else:
        *_, height, width = source_shape
        frames = source_shape[-3] if len(source_shape) >= 5 else 1  # noqa: PLR2004

    batch = 1 if vae_slicing else source_shape[0] if len(source_shape) >= 4 else 1  # noqa: PLR2004

    block_out_channels = _resolve_block_out_channels(vae)
    total_bytes = 0
    for level_index, channels in enumerate(block_out_channels):
        downsample_factor = 2**level_index
        level_height = max(height // downsample_factor, 1)
        level_width = max(width // downsample_factor, 1)
        total_bytes += batch * frames * level_height * level_width * channels * element_size
    return total_bytes
