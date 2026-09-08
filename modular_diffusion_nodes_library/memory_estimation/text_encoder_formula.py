"""Fixed activation-memory overhead for text encoders.

Text encoders (CLIP/T5-family) run once per generation (not once per denoising step)
against a short, fixed sequence length (~77-512 tokens), so their activation memory is
negligible against the denoiser's -- a fixed constant is a safe simplification here,
unlike the VAE (see vae_formula.py) whose activation memory scales with pixel-space
resolution.
"""

from __future__ import annotations

# Order-of-magnitude placeholder, not a calibrated number.
TEXT_ENCODER_ACTIVATION_OVERHEAD_BYTES = 200 * 1024 * 1024


def estimate_text_encoder_activation_bytes() -> int:
    return TEXT_ENCODER_ACTIVATION_OVERHEAD_BYTES
