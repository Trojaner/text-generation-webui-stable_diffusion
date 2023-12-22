import html
import re
from ..params import StableDiffusionWebUiExtensionParams


def try_get_description_prompt(
    message: str, params: StableDiffusionWebUiExtensionParams
) -> bool | str:
    """
    Checks if the given message contains any triggers and returns the prompt if it does.
    """

    trigger_regex = params.interactive_mode_input_trigger_regex
    subject_regex = params.interactive_mode_subject_regex
    default_subject = params.interactive_mode_default_subject
    default_description_prompt = params.interactive_mode_description_prompt
    normalized_message = html.unescape(message).strip()

    if not trigger_regex or not re.match(
        trigger_regex, normalized_message, re.IGNORECASE
    ):
        return False

    subject = default_subject

    if subject_regex:
        match = re.match(subject_regex, normalized_message, re.IGNORECASE)
        if match:
            subject = match.group(0) or default_subject

    return default_description_prompt.replace("[subject]", subject)
