# Spike: Pipeline Memory Estimation Node

Status: brainstorming — no plan approved, no code written yet.

## Goal

Add a node that takes a `PipelineArtifact` (already built/loaded) and a latent shape
(`[height, width, num_frames]` — from a noise latent, a VAE-encode output, or an
encode-masked-media output) and estimates peak memory usage **without running the
pipeline**. Must account for the optimization settings already applied when the
pipeline was built (CPU offload mode, VAE tiling/slicing, attention slicing).

Target accuracy: not exact, but within ~1 GB of actual peak usage. Bias errors toward
overestimating rather than underestimating, so the tool never tells a user a config
will fit when it won't.

## Decisions made so far

- **Weight memory is exact, not estimated.** Since the pipeline is already loaded in
  memory when this node runs, walk each submodule's parameters/buffers
  (`numel() * element_size()`). This automatically reflects quantization and dtype
  correctly — no separate accounting needed.
- **Activation memory is analytical**, not calibrated against real GPU profiling runs
  (profiling was considered and rejected for now — see Open Risk below).
- **Shared formula keyed by architecture family**, not one method per driver. Rationale:
  several drivers share the same token-count/attention shape, so per-driver
  implementations would mostly duplicate each other.
- **Offload modes change the weight term's topology, not just placement**:
  - No offload: weight term = sum of all resident components.
  - `model_cpu_offload`: only one top-level component (text encoder / transformer /
    VAE) is GPU-resident at a time — weight term ≈ max of one component's weights
    (needs verifying against accelerate's actual hook swap timing; there may be a
    brief overlap where two components are resident).
  - `sequential_cpu_offload`: GPU-resident set shrinks to roughly one layer's weights
    at a time — weight term becomes small and activation memory dominates.
- **VAE tiling / VAE slicing / attention slicing** are not separate modes — they feed
  into the same formulas as parameters that reduce the effective spatial tile size or
  sequence length.
- **Output is a breakdown**, not a single number: weight memory, activation/attention
  memory, VAE memory, and a total.

## Open risk: analytical-only accuracy

Sub-1GB accuracy without any real-GPU calibration is optimistic. Sources of error an
analytical formula can't capture: CUDA context overhead (varies by driver/GPU), PyTorch
caching-allocator slack, and whether SDPA actually dispatches to a memory-efficient/
flash kernel for a given shape/dtype/hardware combination vs. falling back to an eager
O(N²) kernel. Mitigation agreed: bias the estimate to overestimate (fixed CUDA-context
constant + safety margin) so misses fail safe. If real-world accuracy turns out worse
than ~1GB in practice, revisit the "analytical only" decision and consider calibrating
just the additive overhead constant per family from a small number of measurements.

## Research findings: architecture families

Full research agent report is in the conversation history that produced this doc
(2026-09-04 session). Summary:

Every driver's attention processor is SDPA-based
(`torch.nn.functional.scaled_dot_product_attention`), which PyTorch can dispatch to a
memory-efficient/flash kernel — so **O(N) activation memory is the reasonable default
assumption across all families**, not O(N²). This meaningfully de-risks the
analytical-only approach.

Three families found (not four — no factorized spatiotemporal attention exists in any
currently-integrated video model):

| Family | Drivers | Token count | Attention |
|---|---|---|---|
| **UNet-SDPA** | SDXL | Per-resolution-level (conv-downsampled, not flat H×W) — needs its own formula shape, summed across down/mid/up blocks | SDPA |
| **Image-DiT-SDPA** | SD3, Flux, Flux2, Flux Fill/Kontext, Qwen, Qwen Edit, Z-Image | `(H_lat/patch) × (W_lat/patch)` flat sequence | SDPA, full joint self-attention |
| **Video-DiT-joint-SDPA** | LTX, LTX2, WAN (+ i2v/VACE/Animate), HunyuanVideo1.5, MiniMax H3 | `(T_lat/patch_t) × (H_lat/patch) × (W_lat/patch)` flat sequence | SDPA, full joint spatiotemporal self-attention (no factorized space/time split found) |

Per-driver details (layers/heads/head_dim/patch size), cited to diffusers source paths,
are in the research agent's report — re-run the same research if this doc has aged and
diffusers has been bumped since.

VAE tiling/slicing (`enable_tiling`/`enable_slicing`) is supported uniformly on the base
`AutoencoderKL` and on every video-specific VAE class in use, so the VAE memory formula
can treat tile size as one parameter across all families rather than needing per-family
VAE logic.

### Caveats not yet resolved

1. Video full-joint-attention (no factorization) is a snapshot of the currently pinned
   diffusers version — if a future model introduces factorized space/time attention,
   the Video-DiT bucket needs splitting.
2. Flux Fill/Kontext, Qwen Edit, and WAN VACE/Animate were not individually verified —
   assumed to inherit their base family's transformer config via driver naming, not
   independently confirmed. Spot-check before shipping.

## Next steps (not started)

1. Draft the actual per-family activation-memory formulas (UNet, Image-DiT, Video-DiT),
   including the per-layer term (hidden_dim, heads, head_dim, num_layers) and where the
   CUDA-context/safety-margin constant gets added.
2. Verify accelerate's CPU-offload hook swap timing (does it ever hold two components
   resident briefly during a swap?) to finalize the offload-mode weight-term formula.
3. Spot-check the two unresolved caveats above.
4. Write the Rule 3-style implementation plan (files to add/modify, node UI, where the
   shared family-formula code lives) once the formulas are settled — this is a new
   node, not a variant or new pipeline type, so it doesn't map to either existing
   `.github/skills/add-*` skill; may need its own lightweight plan structure.
