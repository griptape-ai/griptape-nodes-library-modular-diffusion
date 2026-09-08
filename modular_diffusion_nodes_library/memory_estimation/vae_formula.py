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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact


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

    channel_width = vae.config.block_out_channels[-1]
    return batch * frames * height * width * channel_width * element_size
