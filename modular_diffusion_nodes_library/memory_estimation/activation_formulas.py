"""Analytical, per-family denoiser activation-memory formulas.

Every driver's attention processor is SDPA-based (torch.nn.functional.
scaled_dot_product_attention), which PyTorch can dispatch to a memory-efficient/flash
kernel -- so O(N) activation memory (linear in token count) is the reasonable default
across all families, not O(N^2). ACTIVATION_MULTIPLIER_PER_TOKEN is one lumped constant
standing in for QKV projections + FFN intermediate + residual buffers combined; the
target accuracy is ~20% relative, not exact byte accounting, so a single coarse
constant is intentional rather than separately modeled sub-terms.

Deliberately NOT scaled by num_layers/num_transformer_layers: these formulas estimate
inference-time (no_grad) peak activation memory, where each layer's temporary buffers
are freed before the next layer runs -- unlike training, nothing requires every layer's
activations to be simultaneously resident. Scaling by depth was an earlier bug (see
memory_estimation_implementation.md's revision history): it inflated denoiser
activation estimates by roughly the model's layer count (e.g. ~40x for WAN 14B),
swamping every other term.
"""

from __future__ import annotations

from modular_diffusion_nodes_library.memory_estimation.family_registry import UnetLevelConfig

# Order-of-magnitude placeholder, not a calibrated number -- revisit if real-world
# accuracy checks show it needs adjustment (see docs/spikes/memory_estimation_node.md
# "Open risk" section).
ACTIVATION_MULTIPLIER_PER_TOKEN = 8


def _estimate_joint_attention_activation_bytes(
    token_count: int,
    hidden_dim: int,
    element_size: int,
) -> int:
    return token_count * hidden_dim * ACTIVATION_MULTIPLIER_PER_TOKEN * element_size


def estimate_image_dit_sdpa_activation_bytes(
    height_latent: int,
    width_latent: int,
    hidden_dim: int,
    patch_size_spatial: int,
    element_size: int,
) -> int:
    """Image-DiT-SDPA family: flat (H_lat/patch) x (W_lat/patch) sequence, full joint self-attention."""
    token_count = (height_latent // patch_size_spatial) * (width_latent // patch_size_spatial)
    return _estimate_joint_attention_activation_bytes(token_count, hidden_dim, element_size)


def estimate_video_dit_joint_sdpa_activation_bytes(
    num_frames_latent: int,
    height_latent: int,
    width_latent: int,
    hidden_dim: int,
    patch_size_spatial: int,
    patch_size_temporal: int,
    element_size: int,
) -> int:
    """Video-DiT-joint-SDPA family: flat (T_lat/patch_t) x (H_lat/patch) x (W_lat/patch) sequence."""
    token_count = (
        (num_frames_latent // patch_size_temporal)
        * (height_latent // patch_size_spatial)
        * (width_latent // patch_size_spatial)
    )
    return _estimate_joint_attention_activation_bytes(token_count, hidden_dim, element_size)


def estimate_unet_sdpa_activation_bytes(
    levels: list[UnetLevelConfig],
    height_latent: int,
    width_latent: int,
    element_size: int,
) -> int:
    """UNet-SDPA family: per-resolution-level token count (conv-downsampled), summed across levels.

    Summing across levels (not just taking the largest) is deliberate and distinct from
    the num_layers issue above: U-Net skip connections genuinely keep each level's
    encoder feature map resident until the matching decoder level consumes it, so
    multiple *different-shaped* tensors are concurrently live. Repeated
    same-shape transformer layers within a single level are not (hence no
    num_transformer_layers factor here either).
    """
    total_bytes = 0
    for level in levels:
        tokens_at_level = (height_latent // level.downsample_factor) * (width_latent // level.downsample_factor)
        total_bytes += tokens_at_level * level.hidden_dim * ACTIVATION_MULTIPLIER_PER_TOKEN * element_size
    return total_bytes
