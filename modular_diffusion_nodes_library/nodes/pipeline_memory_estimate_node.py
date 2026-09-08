import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value
from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import estimate_pipeline_memory
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.torch_utils import to_human_readable_size

logger = logging.getLogger("modular_diffusers_nodes_library")


class PipelineMemoryEstimateNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                tooltip=(
                    "Loaded diffusion pipeline to estimate. Connect from Pipeline Builder. The pipeline must "
                    "already be built/loaded (e.g. after a Generate Latent node has run) -- this node never "
                    "triggers a pipeline build."
                ),
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="latent",
                input_types=["LatentArtifact"],
                tooltip="Latent whose shape determines the token count / resolution used in the estimate.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.log_params = LogParameter(self)
        self._create_status_parameters(
            result_details_tooltip="Details about the memory estimate.",
            result_details_placeholder="Memory estimate details appear here after execution.",
        )
        self.log_params.add_output_parameters()

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        if param_name == "pipeline":
            value = normalize_diffusion_pipeline_value(value, node_name=self.name)

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        self._clear_execution_status()
        self.log_params.clear_logs()

        pipeline_artifact = normalize_diffusion_pipeline_value(
            self.get_parameter_value("pipeline"), node_name=self.name
        )
        if pipeline_artifact is None:
            self._set_status_results(was_successful=False, result_details="Missing required 'pipeline' input.")
            return

        latent_artifact = self.get_parameter_value("latent")
        if not isinstance(latent_artifact, LatentArtifact):
            self._set_status_results(was_successful=False, result_details="Missing required 'latent' input.")
            return

        if not model_cache.has_pipeline(pipeline_artifact.config_hash):
            self._set_status_results(
                was_successful=False,
                result_details=(
                    "Pipeline is not currently loaded. Run it through a Generate Latent node or the "
                    "Pipeline Builder first -- this node never triggers a pipeline build."
                ),
            )
            return

        pipe = model_cache.get_pipeline(pipeline_artifact.config_hash)

        try:
            estimate = estimate_pipeline_memory(
                pipe, latent_artifact, pipeline_artifact.optimization_kwargs, pipeline_artifact.pipeline_name
            )
        except Exception as e:
            logger.exception("%s: Pipeline memory estimation failed", self.name)
            self.log_params.append_to_logs("Memory estimation failed.\n")
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return

        self.log_params.append_to_logs(
            f"Pipeline: {estimate.pipeline_name} (offload={estimate.offload_mode or 'none'})\n"
        )
        sum_of_components = 0
        for component in estimate.components:
            sum_of_components += component.total_bytes
            line = (
                f"{component.component_name}: weights={to_human_readable_size(component.weight_bytes)}, "
                f"activations={to_human_readable_size(component.activation_bytes)}, "
                f"total={to_human_readable_size(component.total_bytes)}"
            )
            if component.is_estimated:
                line += f" [ESTIMATED: {component.warning}]"
            self.log_params.append_to_logs(line + "\n")

        self.log_params.append_to_logs(
            f"Sum of all components (if all resident at once): {to_human_readable_size(sum_of_components)}\n"
        )
        self.log_params.append_to_logs(
            f"Estimated peak (topology-aware, offload-adjusted): "
            f"{to_human_readable_size(estimate.estimated_peak_bytes)}\n"
        )
        for warning in estimate.warnings:
            self.log_params.append_to_logs(f"Warning: {warning}\n")

        self._set_status_results(
            was_successful=True,
            result_details=(
                f"Estimated peak: {to_human_readable_size(estimate.estimated_peak_bytes)} "
                f"across {len(estimate.components)} components."
            ),
        )
