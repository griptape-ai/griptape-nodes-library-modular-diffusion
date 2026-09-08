import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    DiffusionPipelineArtifact,
    normalize_diffusion_pipeline_value,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.mixins.parameter_connection_preservation_mixin import (
    ParameterConnectionPreservationMixin,
)
from modular_diffusion_nodes_library.parameters.generate_latent_parameters import (
    DiffusionPipelineGenerateLatentParameters,
)
from modular_diffusion_nodes_library.parameters.pipeline_parameters import (
    ModularDiffusionPipelineParameters,
)
from modular_diffusion_nodes_library.utils.dimension_alignment import snap_dimensions
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")


class NoiseLatentNode(ParameterConnectionPreservationMixin, ControlNode):
    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)
        self.pipe_params = ModularDiffusionPipelineParameters(self)
        self.pipe_params.add_input_parameters()

        self.add_seed_parameter()

        self.add_parameter(
            Parameter(
                name="width",
                default_value=1024,
                type="int",
                tooltip="Width in pixels of the latent (will be divided by the VAE scale factor internally).",
            )
        )
        self.add_parameter(
            Parameter(
                name="height",
                default_value=1024,
                type="int",
                tooltip="Height in pixels of the latent (will be divided by the VAE scale factor internally).",
            )
        )
        self.add_parameter(
            Parameter(
                name="num_frames",
                default_value=41,
                type="int",
                tooltip="Number of video frames to generate for. Ignored for image pipelines.",
            )
        )

        self._dimensionality_warning = ParameterMessage(
            name="dimensionality_warning",
            variant="warning",
            title="Dimensionality Warnings",
            value="",
            hide=True,
        )
        self.add_node_element(self._dimensionality_warning)
        self.latent_parameter = DiffusionPipelineGenerateLatentParameters(self)  # type: ignore[reportOptionalMemberAccess]
        self.latent_parameter.add_output_parameters()
        self._reorder_trailing_parameters()
        self._initializing = False

    def add_seed_parameter(self) -> None:
        self._seed_parameter = SeedParameter(self)
        self._seed_parameter.add_input_parameters()

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

        if parameter.name == "pipeline":
            value = normalize_diffusion_pipeline_value(
                value,
                node_name=self.name,
                raise_on_invalid=True,
            )

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        # hide num_frames parameter if the pipeline doesn't produce video
        if param_name == "pipeline":
            latent_pipeline_driver = get_driver_class(self.pipe_params.get_pipeline_class())
            if latent_pipeline_driver and latent_pipeline_driver.produces_video:
                self.show_parameter_by_name("num_frames")
            else:
                self.hide_parameter_by_name("num_frames")
            self._reorder_trailing_parameters()

        fires_reactively = param_name in ("height", "width", "num_frames", "pipeline")
        if initial_setup and fires_reactively:
            self.after_value_set(parameter, value)

    def _reorder_trailing_parameters(self) -> None:
        """Move ``output_latent`` to the end."""
        trailing = ["output_latent"]
        existing = [element.name for element in self.root_ui_element._children]
        head = [name for name in existing if name not in trailing]
        tail = [name for name in trailing if name in existing]
        self.reorder_elements([*head, *tail])

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name in ("height", "width", "num_frames", "pipeline"):
            self._update_compatibility_message(build_if_needed=False)

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the node.

        This is only allowed during the initialisation stage.
        This prevents changes to the pipeline and runtime parameters
        dynamically adding parameters and modifying connections.
        """
        if not self._initializing:
            return

        super().add_parameter(param)

    def validate_before_node_run(self) -> list[Exception] | None:
        result = self.pipe_params.validate_before_node_run()
        if result is not None:
            return result

        dimension_result = self._update_compatibility_message(build_if_needed=True)
        auto_resize = GriptapeNodes.ConfigManager().get_config_value("modular_diffusion_library.enable_auto_resize")
        if dimension_result is not None and not auto_resize and dimension_result.message:
            return [ValueError(dimension_result.message)]
        return None

    def _update_compatibility_message(self, *, build_if_needed: bool = False):
        pipeline_value = self.get_parameter_value("pipeline")
        if pipeline_value is None or not isinstance(pipeline_value, DiffusionPipelineArtifact):
            self._set_compatibility_message(None)
            return None

        pipeline_class = self.pipe_params.get_pipeline_class()
        if get_driver_class(pipeline_class) is None:
            self._set_compatibility_message(None)
            return None

        if not build_if_needed:
            if not pipeline_value.config_hash or not model_cache.has_pipeline(pipeline_value.config_hash):
                self._set_compatibility_message(None)
                return None

        pipe = self.pipe_params.get_pipeline()
        latent_pipeline_driver = create_driver(pipe, pipeline_class)
        num_frames = self.get_parameter_value("num_frames") or 1
        height = self.get_parameter_value("height") or 1
        width = self.get_parameter_value("width") or 1
        result = snap_dimensions(latent_pipeline_driver, height, width, num_frames)
        self._set_compatibility_message(result.message)
        return result

    def _set_compatibility_message(self, message_str: str | None) -> None:
        if message_str:
            self._dimensionality_warning.value = message_str
            self._dimensionality_warning.hide = False
        else:
            self._dimensionality_warning.value = ""
            self._dimensionality_warning.hide = True

    def preprocess(self) -> None:
        pass

    def process(self) -> AsyncResult:
        self.preprocess()

        def work() -> Any:
            try:
                latent_artifact = self._process()
                self.publish_update_to_parameter("output_latent", latent_artifact)
                self.set_parameter_value("output_latent", latent_artifact)
                self.parameter_output_values["output_latent"] = latent_artifact

            except Exception:
                logger.exception("%s: Diffusion Pipeline execution failed", self.name)
                # Aggressive cleanup on failure
                cleanup_memory_caches()
                raise

        yield work

    def _process(self) -> LatentArtifact:
        pipe = self.pipe_params.get_pipeline()
        latent_pipeline_driver = create_driver(pipe, self.pipe_params.get_pipeline_class())
        height = self.get_parameter_value("height")
        width = self.get_parameter_value("width")
        seed = self.get_parameter_value("seed") or 0
        generator_state = GeneratorState.from_seed(seed)
        num_frames = None
        if latent_pipeline_driver.produces_video:
            num_frames = self.get_parameter_value("num_frames") or None

        result = snap_dimensions(latent_pipeline_driver, height, width, num_frames)
        if result.message:
            logger.warning(result.message)
        height = result.height
        width = result.width
        num_frames = result.num_frames

        if num_frames is not None:
            latents_source_shape = (1, 3, num_frames, height, width)
        else:
            latents_source_shape = (1, 3, height, width)
        return latent_pipeline_driver.create_noise_latent(latents_source_shape, generator_state)
