import threading
from dataclasses import dataclass
from typing import cast
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


# Create a thread-local state for multi-threading support in case
# multiple sessions run concurrently at the same time.
_local_state = threading.local()


def get_current_context() -> GenerationContext | None:
    """
    Gets the current generation context (thread-safe).
    """

    global _local_state
    _local_state.current_context = getattr(_local_state, "current_context", None)
    return cast(GenerationContext | None, _local_state.current_context)


def set_current_context(context: GenerationContext | None) -> None:
    """
    Sets the current generation context (thread-safe).
    """

    global _local_state
    _local_state.current_context = context
