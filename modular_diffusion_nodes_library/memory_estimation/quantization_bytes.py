"""Fixed, known bytes-per-element widths for dynamic quantization / layerwise casting.

Unlike the activation-memory formulas, these are not empirical: `_quantize_diffusion_
pipeline` (pipeline_utils.py:140-155) uses `optimum.quanto`'s qfloat8/qint8/qint4 modes,
and transformer layerwise casting (pipeline_utils.py:338-341) always casts to
torch.float8_e4m3fn -- all fixed, named dtypes, so the resulting bytes-per-element is
deterministic rather than calibrated. See
docs/spikes/memory_estimation_preload_api_plan.md, Decision 3.

Both are gated pipeline-wide by `is_prequantized` / `supports_layerwise_casting`,
mirroring `_manual_optimize_diffusion_pipeline` (pipeline_utils.py:305-341) exactly:
quantization_mode scopes to every real pipeline component; layerwise casting scopes to
the "transformer" slot only (`hasattr(pipe, "transformer")`, pipeline_utils.py:331).
"""

from __future__ import annotations

QUANTIZATION_MODE_BYTES_PER_ELEMENT: dict[str, float] = {
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

LAYERWISE_CASTING_BYTES_PER_ELEMENT = 1.0  # torch.float8_e4m3fn


def resolve_effective_bytes_per_element(
    stored_bytes_per_element: float,
    *,
    slot: str,
    quantization_mode: str,
    transformer_layerwise_casting: bool,
    is_prequantized: bool,
    supports_layerwise_casting: bool,
) -> float:
    """Return the bytes-per-element this component will actually be resident at,
    after applying whatever `_manual_optimize_diffusion_pipeline` would do to it.

    Quantization_mode is checked first (applies to every component), then layerwise
    casting on top (applies to the "transformer" slot only) -- same order and guards
    as the real optimize step. Both are no-ops when `is_prequantized` is True.
    """
    if is_prequantized:
        return stored_bytes_per_element

    effective_bytes_per_element = stored_bytes_per_element
    target_bytes_per_element = QUANTIZATION_MODE_BYTES_PER_ELEMENT.get(quantization_mode)
    if target_bytes_per_element is not None:
        effective_bytes_per_element = target_bytes_per_element

    if transformer_layerwise_casting and slot == "transformer" and supports_layerwise_casting:
        effective_bytes_per_element = LAYERWISE_CASTING_BYTES_PER_ELEMENT

    return effective_bytes_per_element
