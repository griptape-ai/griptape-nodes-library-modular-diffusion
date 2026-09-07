from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import override

import torch  # type: ignore[reportMissingImports]
from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTXVideo  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx.modeling_latent_upsampler import (
    LTXLatentUpsamplerModel,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.ltx.pipeline_ltx_latent_upsample import (
    LTXLatentUpsamplePipeline,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import (
    LTX2LatentUpsamplePipeline,  # type: ignore[reportMissingImports]
)
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

logger = logging.getLogger("diffusers_nodes_library")

SPATIAL_UPSAMPLER_SUBFOLDER = "latent_upsampler"
VAE_SUBFOLDER = "vae"


# ------------------------------------------------------------------
# Base implementations of upsampler parameters.
# ------------------------------------------------------------------


class BaseUpsamplerParameters(ABC):
    def __init__(self, node: BaseNode) -> None:
        self._node = node
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=self._model_repo_id(),
            parameter_name="upsampler_model",
        )

    @abstractmethod
    def _model_repo_id(self) -> list[str]: ...

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        return self._model_repo_parameter.validate_before_node_run()

    def upsample(self, latent_artifact: LatentArtifact) -> LatentArtifact:
        cleanup_memory_caches()
        repo_id, revision = self._model_repo_parameter.get_repo_revision()
        device = get_best_device()
        result = self._upsample(latent_artifact, repo_id, revision, device)
        cleanup_memory_caches()
        return result

    @abstractmethod
    def _upsample(
        self, latent_artifact: LatentArtifact, repo_id: str, revision: str, device: torch.device
    ) -> LatentArtifact: ...


# ------------------------------------------------------------------
# LTX2 upsampler parameters.
# ------------------------------------------------------------------


class LTX2UpsamplerParameters(BaseUpsamplerParameters):
    def __init__(self, node: BaseNode) -> None:
        super().__init__(node)

    @override
    def _model_repo_id(self) -> list[str]:
        return ["dg845/LTX-2.3-Spatial-Upsampler-Diffusers", "Lightricks/LTX-2.5-Diffusers"]

    @override
    def _upsample(
        self, latent_artifact: LatentArtifact, repo_id: str, revision: str, device: torch.device
    ) -> LatentArtifact:
        latent_upsampler_model = LTX2LatentUpsamplerModel.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            subfolder=SPATIAL_UPSAMPLER_SUBFOLDER,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        vae = AutoencoderKLLTX2Video.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            subfolder=VAE_SUBFOLDER,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        pipeline = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler_model)
        pipeline.to(device=device)

        latents_tensor = latent_artifact.to_torch().to(dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            upsampled_raw = pipeline(
                latents=latents_tensor,
                latents_normalized=True,
                output_type="latent",
                return_dict=False,
            )[0]

            upsampled_normalized = LTX2Pipeline._normalize_latents(
                upsampled_raw,  # type: ignore[reportArgumentType]
                vae.latents_mean,
                vae.latents_std,
                vae.config.scaling_factor,  # type: ignore[reportUndefinedVariable]
            )

        upsampled_cpu = upsampled_normalized.detach().cpu()
        pipeline.to("cpu")

        source_shape = latent_artifact.source_shape
        upscaled_source_shape = (*source_shape[:-2], source_shape[-2] * 2, source_shape[-1] * 2)
        return LatentArtifact.from_torch(upsampled_cpu, source_shape=upscaled_source_shape)


# ------------------------------------------------------------------
# LTX upsampler parameters.
# ------------------------------------------------------------------


class LTXUpsamplerParameters(BaseUpsamplerParameters):
    def __init__(self, node: BaseNode) -> None:
        super().__init__(node)

    @override
    def _model_repo_id(self) -> list[str]:
        return ["Lightricks/ltxv-spatial-upscaler-0.9.7"]

    @override
    def _upsample(
        self, latent_artifact: LatentArtifact, repo_id: str, revision: str, device: torch.device
    ) -> LatentArtifact:
        latent_upsampler_model = LTXLatentUpsamplerModel.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            subfolder=SPATIAL_UPSAMPLER_SUBFOLDER,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        vae = AutoencoderKLLTXVideo.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            subfolder=VAE_SUBFOLDER,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        pipeline = LTXLatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler_model)
        pipeline.to(device=device)

        source_shape = latent_artifact.source_shape
        latents_tensor = latent_artifact.to_torch().to(dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            # LTXLatentUpsamplePipeline expects normalized input, denormalizes internally,
            # and re-normalizes the output before returning
            upsampled_normalized = pipeline(
                latents=latents_tensor,
                height=source_shape[-2],
                width=source_shape[-1],
                output_type="latent",
                return_dict=False,
            )[0]

        upsampled_cpu = torch.as_tensor(upsampled_normalized).detach().cpu()
        pipeline.to("cpu")

        upscaled_source_shape = (*source_shape[:-2], source_shape[-2] * 2, source_shape[-1] * 2)
        return LatentArtifact.from_torch(upsampled_cpu, source_shape=upscaled_source_shape)


# ------------------------------------------------------------------
# Provider map and factory for upsampler parameters.
# ------------------------------------------------------------------

UPSAMPLER_TYPE_MAP: dict[str, type[BaseUpsamplerParameters]] = {
    "LTX": LTXUpsamplerParameters,
    "LTX2": LTX2UpsamplerParameters,
}


def create_upsampler_params(upsampler_type: str, node: BaseNode) -> BaseUpsamplerParameters:
    upsampler_cls = UPSAMPLER_TYPE_MAP.get(upsampler_type)
    if upsampler_cls is None:
        msg = f"Attempted to create upsampler parameters. Failed with upsampler_type='{upsampler_type}' because it is not a known upsampler type. Known types: {list(UPSAMPLER_TYPE_MAP.keys())}"
        raise ValueError(msg)
    return upsampler_cls(node)
