from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.parameters.component_override_parameters import ComponentOverrideParameters
from modular_diffusion_nodes_library.parameters.huggingface_pipeline_parameter import HuggingFacePipelineParameter
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.parameters.providers import Provider
from modular_diffusion_nodes_library.standard_parameters.flux2_klein_parameters import (
    Flux2KleinPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.flux2_parameters import (
    Flux2PipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.flux_fill_parameters import (
    FluxFillPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.flux_kontext_parameters import (
    FluxKontextPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.flux_parameters import (
    FluxPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.hunyuan_video1_5_i2v_parameters import (
    HunyuanVideo15ImageToVideoPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.hunyuan_video1_5_parameters import (
    HunyuanVideo15PipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.ltx2_parameters import LTX2PipelineParameters
from modular_diffusion_nodes_library.standard_parameters.ltx25_parameters import (
    LTX25DistilledPipelineParameters,
    LTX25FullPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.ltx_parameters import (
    LTXPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import (
    MiniMaxH3PipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.qwen_edit_parameters import (
    QwenEditPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.qwen_parameters import (
    QwenPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.stable_diffusion_3_parameters import (
    StableDiffusion3PipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.stable_diffusion_sdxl_parameters import (
    StableDiffusionXLPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.wan_animate_parameters import (
    WanAnimatePipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.wan_i2v_parameters import (
    WanImageToVideoPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.wan_parameters import (
    WanPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.wan_vace_parameters import (
    WanVacePipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.z_image_parameters import (
    ZImagePipelineParameters,
)

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )
    from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
        ModularDiffusionPipelineTypePipelineParameters,
    )

logger = logging.getLogger("modular_diffusers_nodes_library")

# This code was copied from diffusers_nodes_library/common/parameters/diffusion/diffusion_pipeline_type_parameters.py.


class LatentPipelineTypeParameters(ABC):
    START_PARAMS: ClassVar = ["pipeline", "provider", "pipeline_type"]
    END_PARAMS: ClassVar = ["loras", "Status", "logs"]

    def __init__(self, node: LatentDiffusionPipelineBuilderNode):
        self._node = node
        self.did_pipeline_type_change = False
        self._pipeline_type_pipeline_params: ModularDiffusionPipelineTypePipelineParameters | None = None
        self.set_pipeline_type_pipeline_params(self.pipeline_types[0])

    @classmethod
    @abstractmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        raise NotImplementedError

    @property
    def pipeline_type_dict(self) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return self.__class__.get_pipeline_type_dict()

    @property
    def pipeline_types(self) -> list[str]:
        return list(self.pipeline_type_dict.keys())

    def set_pipeline_type_pipeline_params(self, pipeline_type: str) -> None:
        try:
            self._pipeline_type_pipeline_params = self.pipeline_type_dict[pipeline_type](self._node)
        except KeyError as e:
            msg = f"Unsupported pipeline type: {pipeline_type}"
            logger.error(msg)
            raise ValueError(msg) from e

    @property
    def pipeline_type_pipeline_params(self) -> ModularDiffusionPipelineTypePipelineParameters:
        if self._pipeline_type_pipeline_params is None:
            msg = "Pipeline type builder parameters not initialized. Ensure provider parameter is set."
            logger.error(msg)
            raise ValueError(msg)
        return self._pipeline_type_pipeline_params

    @property
    def pipeline_type_badge_message(self) -> str:
        """Per-provider badge message shown on the pipeline_type parameter. Override in each subclass."""
        return "Select a pipeline variant. Each variant loads different model weights."

    def add_input_parameters(self) -> None:
        pipeline_type_param = Parameter(
            name="pipeline_type",
            type="str",
            traits={Options(choices=self.pipeline_types)},
            tooltip="Specific pipeline variant within the selected provider (e.g. base, Fill, Edit). Determines which checkpoints and runtime parameters are exposed.",
            allowed_modes={ParameterMode.PROPERTY},
        )
        pipeline_type_param.set_badge(
            variant="help",
            title="Pipeline variants",
            message=self.pipeline_type_badge_message,
        )
        self._node.add_parameter(pipeline_type_param)

    def remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("pipeline_type")
        self.pipeline_type_pipeline_params.remove_input_parameters()

    def before_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "pipeline_type":
            current_pipeline_type = self._node.get_parameter_value("pipeline_type")
            self.did_pipeline_type_change = current_pipeline_type != value

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "pipeline_type" and self.did_pipeline_type_change:
            self.regenerate_elements_for_pipeline_type(value)

    def regenerate_elements_for_pipeline_type(self, pipeline_type: str) -> None:
        self._node.save_parameter_properties()

        self.pipeline_type_pipeline_params.remove_input_parameters()
        self.set_pipeline_type_pipeline_params(pipeline_type)
        self.pipeline_type_pipeline_params.add_input_parameters()

        # Get all current element names
        all_element_names = [element.name for element in self._node.root_ui_element.children]

        # Build parameter groupings
        hf_param_names = HuggingFacePipelineParameter.get_hf_pipeline_parameter_names()

        overrides_group = (
            [ComponentOverrideParameters.GROUP_NAME]
            if ComponentOverrideParameters.GROUP_NAME in all_element_names
            else []
        )
        start_params = LatentPipelineTypeParameters.START_PARAMS
        end_params = [*hf_param_names, *overrides_group, *LatentPipelineTypeParameters.END_PARAMS]
        excluded_params = {*start_params, *end_params}

        # Assemble final order: start -> middle -> end
        middle_params = [name for name in all_element_names if name not in excluded_params]
        sorted_parameters = [*start_params, *middle_params, *end_params]

        self._node.reorder_elements(sorted_parameters)

        self._node.clear_parameter_cache()

    def get_config_kwargs(self) -> dict:
        return self.pipeline_type_pipeline_params.get_config_kwargs()


class LatentFluxPipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `FluxPipeline` — Standard text-to-image generation (FLUX.1-schnell or FLUX.1-dev). "
            "Supports ***ControlNet***, ***inpainting***, and ***ControlNet + inpainting*** via the "
            "ControlNet Pipeline Builder and VAE Mask Encode nodes.\n"
            "- `FluxFillPipeline` — Dedicated inpainting pipeline. Requires a separate Fill checkpoint "
            "(e.g. `FLUX.1-Fill-dev`). Cannot be used with base model weights."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "FluxPipeline": FluxPipelineParameters,
            "FluxFillPipeline": FluxFillPipelineParameters,
            "FluxKontextPipeline": FluxKontextPipelineParameters,
        }


class LatentFlux2PipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `Flux2Pipeline` — Standard text-to-image generation.\n"
            "- `Flux2KleinPipeline` — Guided inpainting with reference image conditioning. "
            "Requires a dedicated Klein checkpoint (4B or 9B). Cannot be used with base Flux2 weights."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "Flux2Pipeline": Flux2PipelineParameters,
            "Flux2KleinPipeline": Flux2KleinPipelineParameters,
        }


class LatentHunyuanVideo15PipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `HunyuanVideo15Pipeline` — Text-to-video generation (Tencent HunyuanVideo 1.5).\n"
            "- `HunyuanVideo15ImageToVideoPipeline` — Image-to-video. Requires a dedicated I2V checkpoint. \n\n"
            "For first-frame conditioning, connect a Media Gen Conditioning node to the "
            "`conditioning_images` input of Generate Media Latents."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "HunyuanVideo15Pipeline": HunyuanVideo15PipelineParameters,
            "HunyuanVideo15ImageToVideoPipeline": HunyuanVideo15ImageToVideoPipelineParameters,
        }


class LatentLTX2PipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `LTX2Pipeline` — Text-to-video and image-to-video generation (Lightricks LTX-Video 2.x).\n\n"
            "Supports HDR output via the Decode HDR node. "
            "Frame count must be a multiple of 8, plus 1 (e.g. 9, 17, 25, 33, 41…).\n\n"
            "- `LTX-2.5 Distilled` / `LTX-2.5 Full (SFT)` — Same `LTX2Pipeline`, built from the single gated "
            "`Lightricks/LTX-2.5-Diffusers` repo. Distilled loads the `transformer` subfolder; Full (SFT) loads "
            "`transformer_full`. Both always decode through the plain vae."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "LTX2Pipeline": LTX2PipelineParameters,
            "LTX-2.5 Distilled": LTX25DistilledPipelineParameters,
            "LTX-2.5 Full (SFT)": LTX25FullPipelineParameters,
        }


class LatentMiniMaxH3PipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `MiniMaxH3ModularPipeline` — Text-to-video and keyframe-to-video generation with a "
            "**jointly generated soundtrack** (MiniMax-H3).\n\n"
            "Video and audio come out of one denoising loop, and the Decode Media Latent node muxes "
            "them into a single MP4. The audio latent travels in the latent's metadata, so connect "
            "Generate Media Latents **directly** to Decode Media Latent — latent math, composite, "
            "upsampler and save/load nodes drop it.\n\n"
            "Fixed 24 fps, 5 to 15 seconds. Frame count is snapped up to the next `17 * n + 5` "
            "(124, 141, 158, … 345). Height and width must be multiples of 32 and default to "
            "MiniMax-H3's own canvas. Guidance is baked into the weights, so there is no "
            "`guidance_scale` and no `negative_prompt`. Image-to-video, video-to-video, ControlNet "
            "and inpainting are not supported."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "MiniMaxH3ModularPipeline": MiniMaxH3PipelineParameters,
        }


class LatentQwenPipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `QwenImagePipeline` — Text-to-image generation. "
            "Supports ***ControlNet***, ***inpainting***, and ***ControlNet + inpainting*** via the "
            "ControlNet Pipeline Builder and VAE Mask Encode nodes.\n"
            "- `QwenImageEditPipeline` — Image editing conditioned on an input image and a text instruction. "
            "Requires a dedicated Edit checkpoint. Cannot be used with base Qwen weights. "
            "Does not support ControlNet."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "QwenImagePipeline": QwenPipelineParameters,
            "QwenImageEditPipeline": QwenEditPipelineParameters,
        }


class LatentStableDiffusionPipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `StableDiffusionXLPipeline` — Text-to-image generation.\n\n"
            "Supports ***ControlNet***, ***inpainting***, and ***ControlNet + inpainting*** via the "
            "ControlNet Pipeline Builder and VAE Mask Encode nodes."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "StableDiffusionXLPipeline": StableDiffusionXLPipelineParameters,
        }


class LatentStableDiffusion3PipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `StableDiffusion3Pipeline` — Text-to-image generation.\n\n"
            "Supports ***ControlNet***, ***inpainting***, and ***ControlNet + inpainting*** via the "
            "ControlNet Pipeline Builder and VAE Mask Encode nodes."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "StableDiffusion3Pipeline": StableDiffusion3PipelineParameters,
        }


class LatentLTXPipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `LTXPipeline` — Text-to-video and image-to-video generation (Lightricks LTX-Video 1.x).\n\n"
            "Frame count must be a multiple of 8, plus 1 (e.g. 9, 17, 25, 33, 41…). "
            "For HDR output or the latest model, use the LTX2 provider instead."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "LTXPipeline": LTXPipelineParameters,
        }


class LatentWanPipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `WanPipeline` — Text-to-video generation (Alibaba WAN).\n"
            "- `WanImageToVideoPipeline` — Image-to-video. Requires a dedicated I2V checkpoint. "
            "Cannot be used with base WAN weights.\n\n"
            "For first/last-frame conditioning, connect a Media Gen Conditioning node to the "
            "`conditioning_images` input of Generate Media Latents."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "WanPipeline": WanPipelineParameters,
            "WanAnimatePipeline": WanAnimatePipelineParameters,
            "WanImageToVideoPipeline": WanImageToVideoPipelineParameters,
            "WanVACEPipeline": WanVacePipelineParameters,
        }


class LatentZImagePipelineTypeParameters(LatentPipelineTypeParameters):
    @property
    def pipeline_type_badge_message(self) -> str:
        return (
            "- `ZImagePipeline` — Text-to-image generation.\n\n"
            "Supports ***ControlNet***, ***inpainting***, and ***ControlNet + inpainting*** via the "
            "ControlNet Pipeline Builder and VAE Mask Encode nodes."
        )

    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "ZImagePipeline": ZImagePipelineParameters,
        }


MODULAR_PIPELINE_TYPE_PROVIDER_MAP: dict[Provider, type[LatentPipelineTypeParameters]] = {
    Provider.FLUX: LatentFluxPipelineTypeParameters,
    Provider.FLUX2: LatentFlux2PipelineTypeParameters,
    Provider.HUNYUAN_VIDEO_1_5: LatentHunyuanVideo15PipelineTypeParameters,
    Provider.LTX: LatentLTXPipelineTypeParameters,
    Provider.LTX2: LatentLTX2PipelineTypeParameters,
    Provider.MINIMAX_H3: LatentMiniMaxH3PipelineTypeParameters,
    Provider.QWEN: LatentQwenPipelineTypeParameters,
    Provider.STABLE_DIFFUSION: LatentStableDiffusionPipelineTypeParameters,
    Provider.STABLE_DIFFUSION_3: LatentStableDiffusion3PipelineTypeParameters,
    Provider.WAN: LatentWanPipelineTypeParameters,
    Provider.Z_IMAGE: LatentZImagePipelineTypeParameters,
}

# Import-time invariant: every registered pipeline class must have all of its
# truly-required __init__ components either exposed as override ports
# (ALLOWED_COMPONENT_SLOTS) or auto-supplied by the params class
# (get_auto_supplied_components()), so the builder can construct a
# fully-overridden pipeline without touching a repo.
for _params_cls in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.values():
    for _pipeline_type_cls in _params_cls.get_pipeline_type_dict().values():
        if _pipeline_type_cls.supports_build_from_overrides_only():
            _pipeline_type_cls.verify_overridable_covers_required()


def find_provider_for_pipeline_type(pipeline_type: str) -> str | None:
    for provider, params_cls in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.items():
        if pipeline_type in params_cls.get_pipeline_type_dict():
            return provider
    return None
