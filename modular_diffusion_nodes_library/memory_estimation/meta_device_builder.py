"""Build a diffusers/transformers component on the meta device from its resolved
config, touching no weight data at all.

Used for pre-load ("not yet built") pipeline memory estimation -- see
docs/spikes/memory_estimation_preload_api_plan.md, Decision 2. This never triggers a
download: config resolution only ever reads the warm HuggingFace cache.

A meta-built component has the same real `.config` object a live one does, so every
existing `family_registry.py` adapter and `torch_utils.get_model_memory()` work on it
unchanged -- no separate formula code needed for the pre-load path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch  # type: ignore[reportMissingImports]
from accelerate import init_empty_weights  # type: ignore[reportMissingImports]
from huggingface_hub import try_to_load_from_cache

from modular_diffusion_nodes_library.component_loading.config_resolver import (
    resolve_hf_repo_config_subfolder,
    try_load_json_dict,
)


class ComponentConfigNotCachedError(RuntimeError):
    """Raised when a component's config file is not in the warm HuggingFace cache."""


def resolve_base_component_config(
    repo_id: str,
    component: str,
    revision: str | None = None,
) -> dict[str, Any]:
    """Read a base-pipeline component's config.json from the warm HF cache.

    Never triggers a download -- raises ComponentConfigNotCachedError if the config
    isn't already cached locally.
    """
    subfolder = resolve_hf_repo_config_subfolder(repo_id, component, component, revision=revision)
    if subfolder is None:
        msg = (
            f"Attempted to resolve config for component '{component}'. "
            f"Failed because no cached config.json was found under repo '{repo_id}' (revision='{revision}')."
        )
        raise ComponentConfigNotCachedError(msg)

    filename = f"{subfolder}/config.json" if subfolder else "config.json"
    cached = try_to_load_from_cache(repo_id, filename=filename, revision=revision)
    if not isinstance(cached, str):
        msg = (
            f"Attempted to resolve config for component '{component}'. "
            f"Failed because '{filename}' is not in the local HuggingFace cache for repo '{repo_id}'."
        )
        raise ComponentConfigNotCachedError(msg)

    config_dict = try_load_json_dict(Path(cached))
    if config_dict is None:
        msg = (
            f"Attempted to resolve config for component '{component}'. "
            f"Failed to parse cached config file '{cached}' as JSON."
        )
        raise ComponentConfigNotCachedError(msg)
    return config_dict


def build_component_on_meta_device(component_cls: type, config_dict: dict[str, Any], torch_dtype: torch.dtype) -> Any:
    """Construct component_cls on the meta device from config_dict, cast to torch_dtype.

    No weight data is ever read or allocated -- shapes/dtype only, via
    accelerate.init_empty_weights(). Dispatches on whether component_cls is a
    transformers PreTrainedModel (constructed from a `config_class` instance) or a
    diffusers ConfigMixin (whose `from_config` accepts a raw dict directly).
    """
    config_class = getattr(component_cls, "config_class", None)
    with init_empty_weights():
        if config_class is not None:
            config_obj = config_class.from_dict(config_dict)
            component = component_cls(config_obj)
        else:
            component = component_cls.from_config(config_dict)
    return component.to(dtype=torch_dtype)
