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

    if not re.match(trigger_regex, message, re.IGNORECASE):
        return False

    subject = default_subject
    match = re.match(subject_regex, message, re.IGNORECASE)
    if match:
        subject = match.group(1)

    return default_description_prompt.replace("[subject]", subject)
