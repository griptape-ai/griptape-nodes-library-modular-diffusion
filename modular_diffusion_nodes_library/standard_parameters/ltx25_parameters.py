import logging
from typing import Any, ClassVar

import torch  # type: ignore[reportMissingImports]
from diffusers.models.transformers.transformer_ltx2 import (  # type: ignore[reportMissingImports]
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.standard_parameters.ltx2_parameters import LTX2PipelineParameters

logger = logging.getLogger("diffusers_nodes_library")


class _LTX25PipelineParametersBase(LTX2PipelineParameters):
    """Shared plumbing for the two LTX-2.5 pipeline_type entries.

    Both entries build an `LTX2Pipeline` from the same repo, differing only in which
    transformer subfolder is loaded and whether `is_distilled` runtime behavior is used.
    Decoding always uses the diffusion decoder (`diffusion_decoder` subfolder), attached to
    the built pipe as a plain attribute; see `LTX2PipelineDriver.decode_latent`.
    """

    _repo_ids: ClassVar[list[str]] = ["Lightricks/LTX-2.5-Diffusers"]
    _transformer_subfolder: str
    _is_distilled: bool

    def get_build_data(self) -> dict[str, Any]:
        build_data = super().get_build_data()
        build_data["is_distilled"] = self._is_distilled
        build_data["transformer_subfolder"] = self._transformer_subfolder
        return build_data

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> LTX2Pipeline:
        overrides.setdefault(
            "transformer",
            LTX2VideoTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path=build_data["base_repo_id"],
                subfolder=build_data["transformer_subfolder"],
                revision=build_data["base_revision"],
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ),
        )
        return super()._build_pipeline_from_repo(build_data, overrides)


class LTX25DistilledPipelineParameters(_LTX25PipelineParametersBase):
    _transformer_subfolder = "transformer"
    _is_distilled = True


class LTX25FullPipelineParameters(_LTX25PipelineParametersBase):
    _transformer_subfolder = "transformer_full"
    _is_distilled = False
