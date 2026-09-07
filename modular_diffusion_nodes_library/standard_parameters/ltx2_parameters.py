import logging
from typing import Any, ClassVar

import torch  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("diffusers_nodes_library")


class LTX2PipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls = LTX2Pipeline
    _repo_ids: ClassVar[list[str]] = [
        "dg845/LTX-2.3-Diffusers",
        "dg845/LTX-2.3-Distilled-Diffusers",
        "Lightricks/LTX-2",
    ]

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        "We don't have support for overriding vocoder and audio vae, so we don't support building from overrides only."
        return False

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):  # noqa: ARG002
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=self._repo_ids,
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        base_repo_id, base_revision = self._resolve_repo(self._model_repo_parameter)

        return {
            "base_repo_id": base_repo_id,
            "base_revision": base_revision,
            "is_distilled": "distilled" in base_repo_id.lower(),
        }

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> LTX2Pipeline:
        return LTX2Pipeline.from_pretrained(
            pretrained_model_name_or_path=build_data["base_repo_id"],
            revision=build_data["base_revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **overrides,
        )
