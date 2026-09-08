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

## Decision 4 — offload topology: requested value; `"Automatic"` is NOT replicated

Post-load: `detect_offload_method(pipe)` — ground truth, live accelerate hooks.

Pre-load, resolved per conversation: **we do not attempt to replicate the `"Automatic"`
cascade.** Rationale (from chat): the primary use case for the pre-load API is "how
much VRAM would this need" — a user asking that question is, in practice, the same
user who will pick a `"Manual"` strategy so the number means something concrete. Not
worth chasing a live-VRAM-dependent decision tree we can't observe.

Correction from initial draft: the Manual/Automatic switch is its own field,
`optimization_kwargs["memory_optimization_strategy"]` (`huggingface_pipeline_parameter.
py:18,29`, choices `["Manual", "Automatic"]`) — not a value of `cpu_offload_strategy`
(whose choices are only `["None", "Model", "Sequential"]`, and which the UI hides
entirely, `after_value_set` at `huggingface_pipeline_parameter.py:131-146`, whenever
`memory_optimization_strategy == "Automatic"`). Dispatch on
`optimization_kwargs["memory_optimization_strategy"]`:
- **`"Manual"`**: use `optimization_kwargs["cpu_offload_strategy"]` directly — exact,
  matches `_manual_optimize_diffusion_pipeline` (`pipeline_utils.py:349-362`)
  one-to-one. `confidence = "high"` for the topology term (activation-formula
  uncertainty still applies on top, same as the post-load path).
- **`"Automatic"`**: treat the real topology as unknown and deliberately pick the
  **conservative bound** — assume no offload, i.e. `peak_weight_bytes = sum of all
  component weight_bytes` (the largest possible resident set). This matches the
  original spike's own bias-toward-overestimate philosophy (`memory_estimation_node.md`
  "Bias errors toward overestimating... so the tool never tells a user a config will
  fit when it won't"): the real Automatic build can only ever reduce memory further via
  offload, never increase it beyond this bound. Attach `confidence = "low"` and a
  warning explaining why, recommending the user switch to Manual for a decision-grade
  number: *"cpu_offload_strategy is 'Automatic'; the actual topology depends on free
  VRAM at build time and cannot be predicted before loading. Showing the upper bound
  (no offload assumed) — actual usage after building will be equal to or lower than
  this. Set the strategy to Manual for an exact pre-load estimate."*

## Decision 5 — component overrides: unify around two orthogonal axes

Confirmed: override descriptors are already reachable off the artifact —
`artifact.build_data.get("_component_overrides", {})` (a `dict[str, ComponentArtifact]`,
same dict `ModularDiffusionPipelineTypePipelineParameters._materialize_overrides()`
consumes, `modular_pipeline_type_parameters.py:133-136`). No new plumbing needed to
reach them.

`ComponentArtifact.try_read_config()` (`component_artifact.py:148-184`) already reads a
component's `config.json` for `HF_REPO`, `LOCAL_DIR`, and `SINGLE_FILE` sources without
materializing — reuse directly.

Rather than branching per `source_type`, split into two orthogonal questions that
apply uniformly to the base pipeline's components AND override components alike:

1. **What are this component's *stored* resident bytes, before any dynamic
   quantization/layerwise-casting is applied?**
   - `HF_REPO` / `LOCAL_DIR` sources (base pipeline components, and `HF_REPO`/
     `LOCAL_DIR` overrides): meta-device + config path (Decision 2).
   - `SINGLE_FILE` sources (override only): the file is unsharded, so
     `Path(file_path).stat().st_size` directly — no meta-device construction needed at
     all, GGUF or not, since there's exactly one file and no sharding/index to resolve.
2. **Does this component get dynamically re-quantized/layerwise-cast on top of its
   stored bytes?**
   - GGUF single-file overrides: **no** — `ComponentArtifact.is_quantized`
     (`component_artifact.py:94-96`) is already `True`, and the builder explicitly
     exempts quantized overrides from layerwise casting (`supports_layerwise_casting =
     ... and not override_is_quantized`, `latent_diffusion_pipeline_builder_node.py:
     172-177`). Stored bytes = final bytes, full stop.
   - Everything else (base components, and non-GGUF overrides — plain `.safetensors`
     single files, `HF_REPO`, `LOCAL_DIR`): **yes, if requested** — a plain-safetensors
     override is still a normal component from `optimize_diffusion_pipeline()`'s point
     of view, so it's still in scope for `quantization_mode` (`get_pipeline_component_
     names(pipe)` iterates all real components, `pipeline_utils.py:153`) and, if it's
     the transformer, for layerwise casting. Apply Decision 3's multiplier table on top
     of the stored-bytes number, respecting the artifact's `is_prequantized` /
     `supports_layerwise_casting` flags exactly as the live optimize step would.

This means the GGUF special-case from the original Decision 2 write-up generalizes:
it isn't "GGUF gets file size, everyone else gets meta-device" — it's "single
unsharded files get file size (trivial either way), and only *quantized-on-disk*
components (GGUF today) are exempt from the dynamic-quantization multiplier."

## Decision 6 — output shape: structured dict is mandatory for the API; messages are node-only

Confirmed in chat: the API's return value must be consumable as a dict (per-component
breakdown + an overall total), and it must carry an explicit accuracy/confidence
signal — honestly, since we don't yet have real-GPU calibration to quote a number
against (same open risk as the original spike's activation-formula placeholder). The
node's human-readable log lines are a separate, node-only concern built on top of this
structured output — no change to that separation of responsibilities.

Extend the existing dataclasses (`estimate_types.py`) rather than inventing a parallel
dict-shaped type, and add `to_dict()` to both:

```python
@dataclass(frozen=True)
class ComponentMemoryEstimate:
    component_name: str
    role: str
    weight_bytes: int
    activation_bytes: int
    total_bytes: int
    is_estimated: bool = False
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]: ...   # new


@dataclass(frozen=True)
class PipelineMemoryEstimate:
    pipeline_name: str
    offload_mode: str | None
    components: list[ComponentMemoryEstimate]
    estimated_peak_bytes: int
    basis: str            # new: "loaded" | "config_only"
    confidence: str       # new: qualitative, e.g. "high" | "low" — see Decision 4
                           # for when "low" applies (Automatic offload today; more
                           # cases may earn it later, e.g. unregistered families)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]: ...   # new — includes a "components" list of
                                                 # each component's to_dict(), plus a
                                                 # top-level total (estimated_peak_bytes)
```

`confidence` is deliberately a coarse qualitative label, not a fabricated percentage —
we don't have calibration data to justify a number (same caveat as `known gaps` #3 in
`memory_estimation_implementation.md`). It goes `"low"` today only for the
`"Automatic"`-offload case; `is_estimated`/`warning` on individual components already
cover per-component caveats (e.g. an unregistered family falling back to weights-only).
`basis` records which of the two code paths (`estimate_pipeline_memory` vs.
`estimate_pipeline_memory_from_build_data`) produced the result, so a caller can tell
which accuracy profile applies without re-deriving it.

The node's `_process()` keeps building its log lines from these fields (unchanged
pattern — see `pipeline_memory_estimate_node.py:111-134`), just adding a line for
`basis`/`confidence` and surfacing `estimated_peak_bytes` and per-component
`to_dict()` output as the node's structured result if the node framework supports a
dict-valued output parameter (needs checking against `SuccessFailureNode` conventions
during implementation — the log-string output stays regardless).

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

## Resolved questions (from 2026-09-08 review)

1. **Weight-file presence gating — not needed.** No new "are weights cached" check.
   Rationale: by the time a `DiffusionPipelineArtifact` exists with a real repo
   selected, the model picker/build flow already implies the weights are locally
   available (the normal build path would error upstream otherwise) — the pre-load
   estimator doesn't need to duplicate that guarantee.
2. **`"Automatic"` offload — not replicated.** See Decision 4: conservative upper-bound
   (no-offload topology) plus an explicit `confidence = "low"` and warning, not a
   cascade simulation.
3. **Output shape — resolved as Decision 6**: dict-convertible dataclasses are
   mandatory for the API; `basis` + `confidence` fields communicate accuracy
   qualitatively; human-readable messages stay a node-only concern layered on top.
4. **Overrides — resolved as Decision 5**: override descriptors are reachable via
   `artifact.build_data["_component_overrides"]`; unified around "stored bytes" (file
   size for single unsharded files, meta-device+config otherwise) plus "is this
   component subject to dynamic quantization/layerwise-casting" (no for GGUF, yes
   otherwise, same rule as base-pipeline components).

## Resolved questions (round 2, 2026-09-08)

5. **Node output UX — keep as-is.** The node stays log/`result_details`-only
   (`pipeline_memory_estimate_node.py:111-141`), no new dict-valued output parameter.
   The dict-shaped `to_dict()` output is for API callers (other nodes/code calling
   `estimate_pipeline_memory_from_artifact()` directly), not surfaced through this
   node's own parameters.
6. **`confidence` stays two-valued ("high"/"low") for now**, driven solely by offload
   strategy per Decision 4. Component-level `is_estimated`/`warning` remain purely
   informational at the component level and do not downgrade the pipeline-level
   `confidence` — revisit only if real usage shows that's confusing.

No open questions remain. Plan is ready for implementation.

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
