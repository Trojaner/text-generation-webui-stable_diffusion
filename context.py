from dataclasses import dataclass
from .params import StableDiffusionWebUiExtensionParams
from .sd_client import SdWebUIApi


@dataclass
class GenerationContext(object):
    params: StableDiffusionWebUiExtensionParams
    sd_client: SdWebUIApi
    input_text: str | None = None
    output_text: str | None = None
    is_completed: bool = False
    state: dict | None = None


_current_context: GenerationContext | None = None


def get_current_context() -> GenerationContext | None:
    """
    Gets the current generation context. Must be called inside a generation request.
    """

    return _current_context


def set_current_context(context: GenerationContext | None) -> None:
    """
    Sets the current generation context. Must be called inside a generation request.
    """

    global _current_context
    _current_context = context
