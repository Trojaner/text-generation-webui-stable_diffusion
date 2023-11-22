from dataclasses import asdict
from typing import Any, List
from modules import shared
from modules.logging_colors import logger
from .context import GenerationContext, get_current_context, set_current_context
from .ext_modules.image_generator import (
    description_to_prompt,
    generate_html_images_for_context,
)
from .ext_modules.text_analyzer import try_get_description_prompt
from .params import StableDiffusionWebUiExtensionParams, TriggerMode
from .sd_client import SdWebUIApi
from .ui import render_ui

ui_params: Any = StableDiffusionWebUiExtensionParams()
ui_params.normalize()

params = asdict(ui_params)

context: GenerationContext | None = None

picture_processing_message = "*Is sending a picture...*"
default_processing_message = shared.processing_message


def chat_input_modifier(text: str, visible_text: str, state: dict) -> [str, str]:
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """

    # bug: this does not trigger on regeneration and hence
    # no context is created in that case

    global context, params, ui_params

    for key in ui_params.__dict__:
        params[key] = ui_params.__dict__[key]

    sd_client = SdWebUIApi(
        baseurl=params["api_endpoint"],
        username=params["api_username"],
        password=params["api_password"],
    )

    input_text = text

    if context is not None and not context.is_completed:
        # A manual trigger was used so only update the context
        context.input_text = input_text
        context.state = state
        context.sd_client = sd_client
        return text, visible_text

    ext_params = StableDiffusionWebUiExtensionParams(**params)

    if ext_params.trigger_mode == TriggerMode.INTERACTIVE:
        description_prompt = try_get_description_prompt(text, ext_params)

        if description_prompt is False:
            # No trigger was found
            return text, visible_text

        assert isinstance(description_prompt, str)
        text = description_prompt

    if ext_params.trigger_mode == TriggerMode.CONTINUOUS:
        # todo: create prompt based on chat history and chat context
        pass

    context = (
        GenerationContext(
            params=ext_params,
            sd_client=sd_client,
            input_text=input_text,
            state=state,
        )
        if context is None or context.is_completed
        else context
    )

    set_current_context(context)

    # todo: check if this can be set in state_modifier instead
    # doesn't seem thread-safe either but it's related to upstream
    shared.processing_message = (
        picture_processing_message
        if context.params.dont_stream_when_generating_images
        else default_processing_message
    )

    return text, visible_text


def state_modifier(state: dict) -> dict:
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """

    context = get_current_context()

    if context is None or context.is_completed:
        return state

    context.state = state

    # bug: no context exists at this point and hence this never works
    # need to initialize context earlier than in chat_input_modifier

    if not context.is_completed and context.params.dont_stream_when_generating_images:
        state["stream"] = False

    return state


def history_modifier(history: List[str]) -> List[str]:
    """
    Modifies the chat history.
    Only used in chat mode.
    """

    context = get_current_context()

    if context is None or context.is_completed:
        return history

    # todo: strip <img> tags from history
    return history


def output_modifier(string: str, state: dict, is_chat: bool = False) -> str:
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """

    context = get_current_context()

    if context is None or context.is_completed:
        # We are currently not generating images, so do nothing
        return string

    context.state = state
    context.output_text = string

    try:
        images_html = generate_html_images_for_context(context)

        if images_html:
            if context.params.trigger_mode == TriggerMode.INTERACTIVE:
                string = f"*{description_to_prompt(string)}*"

            string = f"{images_html}\n{string}"

    except Exception as e:
        string += "\n\n*Image generation has failed.*"
        logger.error(e)
        return string

    context.is_completed = True
    set_current_context(None)
    shared.processing_message = default_processing_message

    return string


def ui() -> None:
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """

    global ui_params

    ui_params = StableDiffusionWebUiExtensionParams(**params)
    render_ui(ui_params)
