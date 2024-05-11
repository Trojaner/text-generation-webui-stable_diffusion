from enum import Enum
from modules.models import load_model, unload_model
from ..context import GenerationContext
import modules.shared as shared

loaded_model = "None"

class VramReallocationTarget(Enum):
    """
    Defines the target for VRAM reallocation.
    """

    STABLE_DIFFUSION = 1
    LLM = 2


def attempt_vram_reallocation(
    target: VramReallocationTarget, context: GenerationContext
) -> None:
    """
    Reallocates VRAM for the given target if dynamic VRAM reallocations are enabled.
    """

    if not context.params.dynamic_vram_reallocation_enabled:
        return

    _reallocate_vram_for_target(target, context)


def _reallocate_vram_for_target(
    target: VramReallocationTarget, context: GenerationContext
) -> None:
    match target:
        case VramReallocationTarget.STABLE_DIFFUSION:
            _allocate_vram_for_stable_diffusion(context)
        case VramReallocationTarget.LLM:
            _allocate_vram_for_llm(context)
        case _:
            raise ValueError(f"Invalid VRAM reallocation target: {target}")


def _allocate_vram_for_stable_diffusion(context: GenerationContext) -> None:
    global loaded_model
    loaded_model = shared.model_name
    unload_model()
    context.sd_client.reload_checkpoint()


def _allocate_vram_for_llm(context: GenerationContext) -> None:
    context.sd_client.unload_checkpoint()
    load_model(loaded_model)
