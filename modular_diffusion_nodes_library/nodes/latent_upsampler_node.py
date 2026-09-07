import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.upsampler_parameter_type import (
    UPSAMPLER_TYPE_MAP,
    BaseUpsamplerParameters,
    create_upsampler_params,
)

logger = logging.getLogger("diffusers_nodes_library")


class LatentUpsamplerNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    START_PARAMS: ClassVar = ["provider"]
    END_PARAMS: ClassVar = ["input_latent", "output_latent", "Status"]

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        self.upsampler_params: BaseUpsamplerParameters | None = None
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="provider",
                default_value="LTX",
                type="str",
                traits={Options(choices=list(UPSAMPLER_TYPE_MAP.keys()))},
                tooltip="Latent upsampler model family (e.g. LTX). Determines which model is loaded.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        default_provider = self.get_parameter_value("provider")
        self.upsampler_params = create_upsampler_params(default_provider, self)
        self.upsampler_params.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="input_latent",
                input_types=["LatentArtifact"],
                type="LatentArtifact",
                tooltip="Latent tensor to upsample to a higher spatial resolution.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_latent",
                output_type="LatentArtifact",
                tooltip="Upsampled latent tensor at a higher spatial resolution.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )
        self._create_status_parameters()
        self._initializing = False

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            return

        if param_name == "provider":
            current_provider = self.get_parameter_value("provider")
            did_provider_change = current_provider != value
        else:
            did_provider_change = False

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if did_provider_change:
            if self.upsampler_params is not None:
                self.upsampler_params.remove_input_parameters()
            self.upsampler_params = create_upsampler_params(value, self)
            self.upsampler_params.add_input_parameters()
            self._reorder_parameters()

    def _reorder_parameters(self) -> None:
        start_params = LatentUpsamplerNode.START_PARAMS
        end_params = LatentUpsamplerNode.END_PARAMS
        excluded_params = {*start_params, *end_params}

        all_element_names = [element.name for element in self.root_ui_element._children]
        middle_params = [name for name in all_element_names if name not in excluded_params]
        sorted_parameters = [*start_params, *middle_params, *end_params]
        self.reorder_elements(sorted_parameters)

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the node.

        Only `upsampler_model` may be added dynamically (on a provider switch) after
        initialization; any other post-init add is dropped. This closes off the path
        that let a stray renamed duplicate (e.g. `upsampler_model_2`) slip through the
        `does_name_exist` guard on a subsequent workflow load.
        """
        is_dynamic_param = param.name == "upsampler_model"
        if not self._initializing and not is_dynamic_param:
            return

        if is_dynamic_param:
            param.user_defined = True

        if not self.does_name_exist(param.name):
            super().add_parameter(param)

    def remove_parameter_element_by_name(self, element_name: str) -> None:
        # HACK: `node.remove_parameter_element_by_name` does not remove connections so we need to use the retained mode request which does.  # noqa: FIX004
        # To avoid updating a ton of callers, we just override this method here.
        # TODO: Remove after https://github.com/griptape-ai/griptape-nodes/issues/2511
        if self.get_element_by_name_and_type(element_name):
            GriptapeNodes.handle_request(
                RemoveParameterFromNodeRequest(parameter_name=element_name, node_name=self.name)
            )

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []

        if self.get_parameter_value("input_latent") is None:
            errors.append(ValueError("Missing required 'input_latent' input."))

        if self.upsampler_params is not None:
            upsampler_errors = self.upsampler_params.validate_before_node_run()
            if upsampler_errors:
                errors.extend(upsampler_errors)
        else:
            errors.append(RuntimeError("Upsampler parameters are not initialized."))

        return errors or None

    def process(self) -> AsyncResult:
        self._clear_execution_status()
        yield lambda: self._run_with_status(
            self._upsample,
            success_msg="Upsampling completed successfully.",
            failure_log="Latent upsampling failed",
            logger=logger,
        )

    def _upsample(self) -> None:
        if self.upsampler_params is None:
            raise RuntimeError(f"{self.name}: upsampler parameters are not initialized.")

        latent_artifact: LatentArtifact = self.get_parameter_value("input_latent")
        output_artifact = self.upsampler_params.upsample(latent_artifact)
        self.set_parameter_value("output_latent", output_artifact)
        self.parameter_output_values["output_latent"] = output_artifact
