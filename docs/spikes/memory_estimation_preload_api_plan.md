# Plan: Dual-Path Memory Estimation API (pre-load + post-load)

Status: **planning — awaiting review, no code written yet.** Extends the implemented
work in `memory_estimation_implementation.md` (post-load path); does not replace it.

## Goal

Two consumable surfaces, per the 2026-09-08 conversation:

1. **An API** (`modular_diffusion_nodes_library/memory_estimation/`) callable from any
   node, given a `DiffusionPipelineArtifact` + a latent shape, that estimates memory
   **regardless of whether the pipeline has actually been built yet**.
2. **The existing node** (`PipelineMemoryEstimateNode`), updated to call this API instead
   of `estimate_pipeline_memory()` directly, so it transparently supports both states.

The dispatch between "pipeline already loaded" and "not loaded" is the pivot of this
plan; see Decision 1 below.

## Decision 1 — how the API tells the two cases apart

`DiffusionPipelineArtifact` never holds the built pipeline (confirmed by reading
`pipeline_artifact.py:60-63` and `__init__`); the loaded object lives only in the
process-global `model_cache`, keyed by `artifact.config_hash`
(`utils/huggingface_utils.py:80`, `ModelCache.has_pipeline` / `get_pipeline`).

So the single public entry point becomes:

```python
def estimate_pipeline_memory_from_artifact(
    artifact: DiffusionPipelineArtifact,
    latent: LatentArtifact,
) -> PipelineMemoryEstimate:
    if model_cache.has_pipeline(artifact.config_hash):
        pipe = model_cache.get_pipeline(artifact.config_hash)
        return estimate_pipeline_memory(          # existing, unchanged
            pipe, latent, artifact.optimization_kwargs, artifact.pipeline_name
        )
    return estimate_pipeline_memory_from_build_data(   # new
        artifact, latent
    )
```

Callers (including the node) only ever call `estimate_pipeline_memory_from_artifact()`.
`estimate_pipeline_memory()` (today's function, exact/post-load) stays as a lower-level
building block, unchanged.

Note this is a **cache-key match**, not "is a pipeline for this repo loaded anywhere" —
if the user changes an optimization setting after building, `config_hash` changes and
`has_pipeline()` correctly reports `False` even though a (differently-configured)
pipeline is resident. That's desired: the estimate should reflect the exact config being
asked about.

## Decision 2 — pre-load weight bytes via `meta` device, not file size

Superseded from earlier brainstorming (file-size-on-disk approach): build each
component on `accelerate.init_empty_weights()` (already a pinned dependency,
`pyproject.toml:10`) from its resolved `config.json`, then reuse `get_model_memory()`
verbatim:

```python
config_dict = try_load_json_dict(resolved_config_path)   # config_resolver.py, exists today
with init_empty_weights():
    component = component_cls.from_config(config_dict)
component = component.to(dtype=target_dtype)             # no-op cost on meta tensors
weight_bytes = get_model_memory(component)                # torch_utils.py, UNCHANGED
```

Why this beats file size:
- `get_model_memory()` and every `family_registry.py` adapter (`_adapt_flux_family`,
  etc.) need **zero changes** — a meta-built component has the same real `.config`
  object a live one does.
- dtype is asserted explicitly, not reverse-engineered from on-disk bytes.
- No need to reimplement HF's safetensors shard/index resolution just to count bytes.
- Needs only `config.json` to be cache-resolvable — not the weight shards. See Open
  Question 1 for the scope implication.

**Exception: GGUF single-file component overrides.** `ComponentArtifact.is_quantized`
(`component_artifact.py:94-96`) is `True` when `file_path` ends in `.gguf`. These are
already stored in their final quantized block layout — reading `Path(file_path).stat().
st_size` directly is more accurate than reconstructing from a meta-built architecture
plus a multiplier, since GGUF block-quantization overhead isn't a clean bytes/element
ratio. This is the one place file size remains the right tool.

## Decision 3 — quantization/layerwise-casting byte-width multiplier

`_quantize_diffusion_pipeline` (`pipeline_utils.py:140-155`) uses `optimum.quanto` with
exactly three modes, each a **fixed, known** bit width (not empirical, unlike the
existing `ACTIVATION_MULTIPLIER_PER_TOKEN` placeholder):

| `quantization_mode` | bytes/element |
|---|---|
| `"None"` | on-disk element size (no change) |
| `"fp8"` | 1 |
| `"int8"` | 1 |
| `"int4"` | 0.5 |

Layerwise casting (`pipeline_utils.py:338-341`) always casts the **transformer only**
to `torch.float8_e4m3fn` → 1 byte/element, scoped to that one component.

`quantization_mode` scope = all components (`get_pipeline_component_names(pipe)`,
`pipeline_utils.py:153`); layerwise casting scope = transformer only. New table lives
in a new module (see Files below), applied as a post-multiply on the meta-computed
`weight_bytes` for affected components. Quanto's per-group scale-factor overhead is
ignored (negligible at GB scale) — worth a one-line comment, not a blocker.

## Decision 4 — offload topology: requested vs. replicated-Automatic

Post-load: `detect_offload_method(pipe)` — ground truth, live accelerate hooks.

Pre-load, two sub-cases from `optimization_kwargs["cpu_offload_strategy"]`:
- **`"None"` / `"Model"` / `"Sequential"` (explicit/Manual)**: use directly — exact,
  no approximation, matches `_manual_optimize_diffusion_pipeline`
  (`pipeline_utils.py:349-362`) one-to-one.
- **`"Automatic"`**: `pipeline_utils.py:208-254` shows the real decision is a live
  cascade — checks `_check_cuda_memory_sufficient`, falls back to fp8 layerwise
  casting, then `model_cpu_offload`, then `sequential_cpu_offload` — all gated on
  `get_free_cuda_memory()` / `get_max_memory_footprint(pipe, ...)` measured against the
  **actually loaded** pipeline. Pre-load, we can only *replicate* this cascade using our
  own meta-derived weight-byte estimate in place of the live one. This is the largest
  structural source of divergence between the two API paths — see the accuracy section
  below.

## Decision 5 — component overrides need no new weight-file resolution logic

`ComponentArtifact.try_read_config()` (`component_artifact.py:148-184`) already reads a
component's `config.json` for `HF_REPO`, `LOCAL_DIR`, and `SINGLE_FILE` sources without
materializing — reuse directly, no new code. For weight bytes:
- `HF_REPO` / `LOCAL_DIR` overrides: same meta-device path as the base pipeline, fed
  the config `try_read_config()` returns.
- `SINGLE_FILE` overrides: `self.file_path` already points at the weight file — GGUF
  uses file size directly (Decision 2's exception); non-GGUF single files (plain
  `.safetensors`) can go through file size too, since there's no separate "requested
  dtype" step for a raw single-file load beyond what's already baked into the file.

## How much will the two APIs differ, in theory?

**In the common case — no dynamic quantization, no layerwise casting, explicit (not
Automatic) offload strategy — they should be numerically identical, or extremely
close to it.** Every value the shared formulas consume is sourced from the same place
either way:

| Input to the shared formulas | Post-load source | Pre-load source | Expected to match? |
|---|---|---|---|
| Denoiser config fields (hidden_dim, num_layers, patch_size) | `pipe.transformer.config` | meta-built `component.config` from the same `config.json` | **Yes, exactly** — same file, same `FrozenDict` shape |
| Weight bytes (unquantized) | `get_model_memory(real component)` | `get_model_memory(meta component)` | **Yes, exactly** — numel × element_size depends only on shape/dtype, both meta and real tensors report these identically |
| VAE tile attrs / channel width | live `vae.config` | meta-built `vae.config` | **Yes, exactly** |
| Activation formula output | same function, same inputs | same function, same inputs | **Yes, exactly** — formula code is untouched either way |
| Offload topology (Manual) | live hook detection | requested value | **Yes, exactly** |

So for that common configuration, any difference between the two estimates would
indicate a **bug** (e.g. a stale cache, a config resolved from the wrong revision), not
expected approximation error — worth asserting as an equality property in tests.

**Where real divergence is expected to enter:**

1. **`"Automatic"` offload strategy — the dominant risk, and it's structural, not a
   rounding error.** The live cascade branches on *actual* free VRAM at the moment the
   real build runs; the pre-load replica can only use free VRAM at *estimate* time, plus
   our own weight-byte estimate instead of the real one as the cascade's input. Two
   failure modes: (a) free VRAM genuinely changes between estimate-time and build-time
   (another process/pipeline claims or frees memory), or (b) the replica doesn't fully
   mirror every branch of `pipeline_utils.py:208-254` (e.g. the fp8-layerwise-casting
   fallback that's tried *before* model/sequential offload). Getting (b) wrong doesn't
   cause a percentage error — it can put the estimate in an entirely different topology
   bucket (e.g. predicting "Sequential" when the real build lands on "Model" after fp8
   casting succeeds), which swings peak weight bytes by potentially several GB, not
   percent. This needs the replica to mirror the cascade exactly, in order, or to be
   explicit in the output that "Automatic" estimates carry materially higher
   uncertainty than "Manual" ones.
2. **Quantization / layerwise-casting multiplier** — quanto's real per-group scale
   overhead is ignored; expect a small, bounded error (low single digits of a percent
   of that component's weight bytes), well inside the existing ~20% relative accuracy
   target on its own. Not a concern in isolation; worth remembering it compounds with
   risk 1 if both apply.
3. **GGUF overrides** — expected to be near-zero divergence; file size already reflects
   final resident bytes.

Net: the plan should surface *which* path produced an estimate (exact/post-load vs.
config-derived/pre-load) and, within the pre-load path, flag `"Automatic"` results as
lower-confidence than `"Manual"` ones, rather than presenting all pre-load numbers with
uniform confidence.

## Files to add / modify

```
modular_diffusion_nodes_library/memory_estimation/
    meta_device_builder.py          (new) — resolve config (reuse config_resolver.py /
                                     try_read_config()), build under init_empty_weights(),
                                     apply target dtype
    quantization_bytes.py           (new) — the fixed bytes/element table (Decision 3)
                                     and which components each mode scopes to
    pipeline_memory_estimator.py    (modify) — add estimate_pipeline_memory_from_artifact()
                                     (dispatch) and estimate_pipeline_memory_from_build_data()
                                     (new, meta-device path); existing
                                     estimate_pipeline_memory() untouched
    family_registry.py              (no change expected — adapters already operate on
                                     any object exposing `.config`)
    activation_formulas.py          (no change)
    vae_formula.py                  (no change to formula body; confirm it also works
                                     against a meta-built vae's `.config`)
    text_encoder_formula.py         (no change)

modular_diffusion_nodes_library/nodes/pipeline_memory_estimate_node.py
    (modify) — call estimate_pipeline_memory_from_artifact() instead of gating on
    model_cache.has_pipeline() itself; update the "not loaded" failure path since it's
    no longer a failure case, and update the log output to state which path ran

tests/memory_estimation/
    test_meta_device_builder.py     (new)
    test_quantization_bytes.py      (new)
    test_pipeline_memory_estimator.py (extend) — cross-path equality test for the
    "common case" table above, plus Automatic-offload divergence coverage
```

## Open questions for review

1. **Should the pre-load path require weight files to be cache-resolvable too, even
   though meta-construction technically only needs `config.json`?** Confirmed in chat:
   yes — gate on weight presence for consistency with the "user only has access to
   already-downloaded pipelines" framing, even though the estimate computation itself
   doesn't strictly need it. Needs a concrete "are the weights cached" check (e.g. probe
   for `model.safetensors` / `model.safetensors.index.json` / a single `.safetensors`
   file via `try_to_load_from_cache`, mirroring `config_resolver.py`'s pattern) that
   fails cleanly with a clear message if absent, rather than silently proceeding.
2. **How faithfully should the `"Automatic"` cascade be replicated?** Options: (a) fully
   mirror every branch in `pipeline_utils.py:208-254` including the fp8-layerwise
   fallback order, or (b) a simplified two-bucket heuristic with an explicit low-
   confidence flag. (a) is more accurate but doubles the surface area that has to stay
   in sync with `pipeline_utils.py` if that logic changes later.
3. **Node output UX**: does the node need a visible "estimate basis: loaded pipeline /
   config-only" line, and for the latter, an "Automatic offload — treat with extra
   caution" flag? (Recommended given the accuracy analysis above.)
4. **Plain single-file (non-GGUF) `.safetensors` overrides** — Decision 5 proposes file
   size for these too (no separate dtype step). Confirm this is right rather than also
   routing them through meta-device + multiplier.

## Verification plan

1. `uv run ruff format --check .` + `uv run ruff check .` — clean.
2. New unit tests for `meta_device_builder.py` / `quantization_bytes.py` — pure,
   synthetic, no GPU/model download (matches existing `tests/memory_estimation/`
   convention).
3. Cross-path equality test: build a small synthetic config through both
   `estimate_pipeline_memory()` (mocked live pipe) and
   `estimate_pipeline_memory_from_build_data()` (meta-built from the same config) and
   assert equal weight_bytes/activation_bytes for the non-quantized/Manual-offload case.
4. Manual smoke test once implemented: build a real small pipeline, estimate it via the
   node (a) before running it (pre-load path) and (b) after running it (post-load path),
   compare the two numbers for a Manual-offload config — should match closely per the
   theory above; any large mismatch is a bug to chase down.
