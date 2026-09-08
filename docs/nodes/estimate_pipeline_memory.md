# Estimate Pipeline Memory

**Estimates per-component GPU memory usage for an already-loaded diffusion pipeline, without running it.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Breaks down memory into weights and activations for each resident component (text encoders, transformer/UNet, VAE).
- Uses the latent's shape to size the denoiser's and VAE's activation-memory estimate, so it must be wired to a real latent.
- Requires the pipeline to already be built/loaded (e.g. after a Generate Media Latents node has run); it never triggers a pipeline build itself.
- Estimates are analytical, biased to overestimate, and target ~20% relative accuracy — not measured, exact byte counts.

## Typical workflow position

```text
Pipeline Builder → Generate Media Latents → [Estimate Pipeline Memory]
```

## Node preview

<!-- TODO: add docs/assets/nodes/estimate-pipeline-memory.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `pipeline` | `Pipeline Config` | Yes | Connect from Pipeline Builder. Must already be loaded/resident in the model cache. |
| `latent` | `LatentArtifact` | Yes | Determines the token count / resolution used in the activation-memory estimate. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `logs` | `str` | Multiline per-component breakdown (weights, activations, total) plus the estimated peak. |
| `was_successful` | `bool` | `True` when the estimate completed without exception. |
| `result_details` | `str` | Summary line with the estimated peak and component count. |

## Provider / model behavior

Denoiser activation memory is estimated using one of three architecture families: UNet-SDPA (SDXL), Image-DiT-SDPA (Flux, Flux2, SD3, Qwen, Z-Image, ...), or Video-DiT-joint-SDPA (LTX, LTX2, WAN, HunyuanVideo 1.5, MiniMax-H3). Weight memory always reflects the pipeline's actual loaded dtype and quantization, since it is read directly off the resident tensors. VAE activation memory reflects whether `vae_tiling`/`vae_slicing` were enabled at build time. A pipeline class without a registered family, or a component whose config can't be read (e.g. a heavily fused/custom checkpoint), falls back to a weight-only estimate with a warning in the logs rather than failing.

## Tips & pitfalls

- **Run this after the pipeline is loaded, not just built.** If the pipeline artifact's config hash isn't resident in the model cache yet, the node fails with a clear message instead of triggering a build.
- **Watch for `[ESTIMATED: ...]` warnings in the logs.** These flag components whose activation memory could not be computed (unrecognized component, unregistered pipeline family, or an unreadable transformer/VAE config) — only weight memory is shown for those.
- **Treat the number as a guide, not a guarantee.** Estimates are analytical (no GPU profiling) and biased to overestimate; expect divergence from true peak usage.

## See also

- [Modular Diffusion Pipeline Builder](pipeline_builder.md)
- [Generate Media Latents](generate_media_latents.md)
- [Clear Pipeline Cache](clear_pipeline_cache.md)
