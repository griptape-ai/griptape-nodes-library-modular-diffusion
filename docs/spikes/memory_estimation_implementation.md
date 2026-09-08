# Memory Estimation: Implementation Notes (2026-09-07)

Status: **implemented**, lint/tests passing. Supersedes the brainstorming in
`memory_estimation_node.md` (kept for historical context/rationale).

## What was built

A per-component GPU memory estimation API plus an informational node, for an
**already-loaded** diffusion pipeline. No pipeline execution required.

### New module: `modular_diffusion_nodes_library/memory_estimation/`

| File | Responsibility |
|---|---|
| `estimate_types.py` | `ComponentMemoryEstimate` / `PipelineMemoryEstimate` dataclasses. `ComponentMemoryEstimate.create()` derives `total_bytes` so it can't drift. |
| `family_registry.py` | `MemoryFamily` enum (UNET_SDPA / IMAGE_DIT_SDPA / VIDEO_DIT_JOINT_SDPA), `_FAMILY_BY_PIPELINE_NAME` map (kept in sync with `driver_factory._DRIVER_REGISTRY` by a unit test), per-pipeline-class denoiser config adapters (`_adapt_flux_family`, `_adapt_ltx_family`, `_adapt_wan_family`, `_adapt_minimax_h3`, etc.), and `get_sdxl_unet_levels()` for the UNet's own per-resolution-level shape. |
| `activation_formulas.py` | Pure formulas for the three denoiser families. Shared core `_estimate_joint_attention_activation_bytes` (token_count × hidden_dim × num_layers × `ACTIVATION_MULTIPLIER_PER_TOKEN` × element_size) backs both Image-DiT and Video-DiT. UNet-SDPA sums per-resolution-level. |
| `vae_formula.py` | Conv/tile-based VAE activation formula, tile/slice-aware via `optimization_kwargs["vae_tiling"/"vae_slicing"]` and `latent.source_shape`. |
| `text_encoder_formula.py` | Fixed activation-overhead constant (`TEXT_ENCODER_ACTIVATION_OVERHEAD_BYTES`, placeholder ~200MB). |
| `pipeline_memory_estimator.py` | Orchestrator: `estimate_pipeline_memory()` (whole pipeline) and `estimate_component_memory()` (single component). Only module touching the real `pipe` object. Handles offload-mode topology and all fallback logic. |

### New node

`modular_diffusion_nodes_library/nodes/pipeline_memory_estimate_node.py` —
`PipelineMemoryEstimateNode`, registered in `griptape-nodes-library.json` as
"Estimate Pipeline Memory" under `ModularDiffusion/Pipeline`.

- Inputs: `pipeline` (`Pipeline Config`), `latent` (`LatentArtifact`).
- Peeks `model_cache.has_pipeline`/`get_pipeline` by `config_hash` — **never** calls
  `get_or_build_pipeline()`. Fails cleanly if the pipeline isn't already resident.
- Outputs a `logs` breakdown (via the standard `LogParameter` pattern) plus the usual
  `was_successful`/`result_details` status parameters.

### Docs

`docs/nodes/estimate_pipeline_memory.md`, linked from `docs/index.md` under Pipeline.

### Tests

`tests/memory_estimation/` — 35 pure unit tests (formulas, adapters, topology logic,
fallback paths), all synthetic/mocked, no GPU or real model download needed.

## Key decisions made during planning (chat history, not repeated in code comments)

1. **Post-load only.** No pre-load/config-only estimation path — the node runs after
   the pipeline builder, so the pipeline is always already loaded.
2. **Per-component list output**, not a single aggregate breakdown.
3. **Accuracy target is ~20% relative**, not a fixed absolute GB margin (this changed
   mid-planning from the original spike's ~1GB target). The sole safety margin is the
   existing `MEMORY_HEADROOM_FACTOR = 1.2` from `utils/pipeline_utils.py` — no separate
   fixed-byte constant.
4. **Text encoders get weight memory + a small fixed activation constant.** VAE gets
   its own tile/slice-aware analytical formula — explicitly *not* a constant, because
   VAE activation memory scales with pixel-space resolution and can swing by multiple
   GB depending on tiling.
5. **No library shortcut exists.** HF `accelerate`'s memory estimator only covers
   weight memory (which this repo already computes via `torch_utils.get_model_memory`)
   plus a flat +20%-of-weights heuristic for "everything else" — wrong for diffusion
   models, where activation memory scales with latent resolution/frame count, largely
   independent of weight size. Confirmed by web research during planning.
6. **Offload mode ground truth is `detect_offload_method(pipe)`** (live accelerate hook
   inspection), not `DiffusionPipelineArtifact.optimization_kwargs["cpu_offload_strategy"]` —
   the latter only reflects the *requested* value under "Manual" strategy and is stale
   under "Automatic".
7. **No new fields needed on `DiffusionPipelineArtifact`.** `pipeline_name`,
   `optimization_kwargs`, `config_hash` were already sufficient. The node does not
   duplicate any work the Pipeline Builder already did — weight/activation math is new
   computation the builder never performed.

## A real bug caught during implementation

`source_shape`/`latent.shape` frame-dimension indexing initially used
`len(shape) >= 3` to detect a temporal dimension. This is wrong: image shapes are 4D
`(B, C, H, W)` and video shapes are 5D `(B, C, T, H, W)` — the `>= 3` check treated an
image's **channel** dimension as a frame count. Fixed to `>= 5` in both
`vae_formula.py` and `pipeline_memory_estimator.py`. Caught by writing the VAE unit
tests, not by inspection — worth remembering if similar shape-indexing code gets added
elsewhere.

## Known gaps / follow-ups (not blocking, but worth revisiting)

1. **Per-pipeline adapter reuse is assumed, not individually verified**, for: Flux
   Fill/Kontext (assumed same transformer class as Flux), Qwen Edit (assumed same as
   Qwen), WAN i2v/VACE/Animate (assumed same as WAN T2V), HunyuanVideo15 i2v (assumed
   same as base). Only Flux, LTX, SDXL, Qwen, WAN, SD3, Flux2, Z-Image, HunyuanVideo15,
   LTX2, and MiniMax-H3 base transformer configs were checked directly against live
   diffusers constructor signatures.
2. **VAE tile-size attribute name** was verified for `AutoencoderKL`
   (`tile_sample_min_size`) and `AutoencoderKLLTXVideo`/`AutoencoderKLWan`
   (`tile_sample_min_height`/`tile_sample_min_width`, LTX also has
   `tile_sample_min_num_frames`) — not exhaustively checked for every VAE class in the
   registry (e.g. HunyuanVideo15, LTX2, MiniMax-H3's VAEs).
3. **`ACTIVATION_MULTIPLIER_PER_TOKEN` (8), `TEXT_ENCODER_ACTIVATION_OVERHEAD_BYTES`
   (~200MB) are order-of-magnitude placeholders**, not calibrated against any real
   measurement. No integration test exists yet against a real loaded pipeline to sanity
   check these against actual GPU memory usage.
4. **Sequential-offload's per-layer weight approximation** (`weight_bytes // num_layers`
   for the largest denoiser component) is a rough approximation, not verified against
   accelerate's actual hook-swap granularity.
5. **No integration test with a real small loaded pipeline exists yet** — this repo's
   `tests/` has no existing fixture/pattern for that (checked during planning; only
   `test_model_catalog_consistency.py` existed beforehand). The plan called for one;
   it wasn't added because there's no established convention to follow, and inventing
   one felt like scope creep beyond what was asked. Worth revisiting explicitly.

## Files touched

```
modular_diffusion_nodes_library/memory_estimation/__init__.py          (new)
modular_diffusion_nodes_library/memory_estimation/estimate_types.py    (new)
modular_diffusion_nodes_library/memory_estimation/family_registry.py   (new)
modular_diffusion_nodes_library/memory_estimation/activation_formulas.py (new)
modular_diffusion_nodes_library/memory_estimation/vae_formula.py       (new)
modular_diffusion_nodes_library/memory_estimation/text_encoder_formula.py (new)
modular_diffusion_nodes_library/memory_estimation/pipeline_memory_estimator.py (new)
modular_diffusion_nodes_library/nodes/pipeline_memory_estimate_node.py (new)
griptape-nodes-library.json                                            (modified: node registration)
docs/nodes/estimate_pipeline_memory.md                                 (new)
docs/index.md                                                          (modified: nav entry)
tests/memory_estimation/*.py                                           (new, 35 tests)
```

The full design plan (with code-shape sketches, rejected alternatives, and the
library-research discussion) is preserved in the Claude Code plan file from this
session: `abstract-crafting-eagle.md` (session-local, not in this repo).

## How to pick this back up

1. `uv run ruff format --check .` + `uv run ruff check .` + `uv run pytest tests/memory_estimation` — should all be clean as of this writing.
2. If continuing: tackle the "Known gaps" list above, roughly in order (adapter
   verification first — it's the easiest correctness win; calibration constants last —
   they need real measurements).
3. If something looks wrong in a specific driver's memory estimate, start by checking
   `family_registry.py`'s adapter for that pipeline class against the live diffusers
   transformer/VAE config — the field names are genuinely inconsistent across model
   families (confirmed by direct introspection during implementation), so a wrong
   number is more likely an adapter bug than a formula bug.
