import base64
import dataclasses
import html
import io
import re
import time
from datetime import date
from pathlib import Path
from typing import cast
from webuiapi import WebUIApiResult
from modules.logging_colors import logger
from ..context import GenerationContext
from ..params import (
    ContinuousModePromptGenerationMode,
    InteractiveModePromptGenerationMode,
    RegexGenerationRuleMatch,
    TriggerMode,
)
from .vram_manager import VramReallocationTarget, attempt_vram_reallocation


def normalize_regex(regex: str) -> str:
    if not regex.startswith("^") and not regex.startswith(".*"):
        regex = f".*{regex}"

    if not regex.endswith("$") and not regex.endswith(".*"):
        regex = f"{regex}.*"

    return regex


def normalize_prompt(prompt: str, do_additional_normalization: bool = False) -> str:
    if prompt is None:
        return ""

    result = (
        prompt.replace("*", "")
        .replace('"', "")
        .replace("!", ",")
        .replace("?", ",")
        .replace("&", "")
        .replace(",,", ",")
        .replace(", ,", ",")
        .replace(";", ",")
        .strip()
        .strip(",")
        .strip()
    )

    # deduplicate tags
    tags = set([x.strip() for x in result.split(",")])
    return ", ".join(tags)


def generate_html_images_for_context(
    context: GenerationContext,
) -> tuple[str | None, str, str, str, str]:
    """
    Generates images for the given context using Stable Diffusion
    and returns the result as HTML output
    """

    attempt_vram_reallocation(VramReallocationTarget.STABLE_DIFFUSION, context)

    sd_client = context.sd_client

    rules_prompt = ""
    rules_negative_prompt = ""

    faceswaplab_force_enabled: bool | None = None
    faceswaplab_overwrite_source_face: str | None = None

    reactor_force_enabled: bool | None = None
    reactor_overwrite_source_face: str | None = None

    if context.params.generation_rules:
        for rule in context.params.generation_rules:
            try:
                match_against = []

                delimiters = ".", ",", "!", "?", "\n", "*", '"'
                delimiters_regex_pattern = "|".join(map(re.escape, delimiters))

                if "match" in rule:
                    if (
                        context.input_text
                        and context.input_text != ""
                        and RegexGenerationRuleMatch.INPUT.value in rule["match"]
                    ):
                        match_against.append(context.input_text.strip())

                    if (
                        context.input_text
                        and context.input_text != ""
                        and RegexGenerationRuleMatch.INPUT_SENTENCE.value
                        in rule["match"]
                    ):
                        match_against += [
                            x.strip()
                            for x in re.split(
                                delimiters_regex_pattern, context.input_text
                            )
                            if x.strip() != ""
                        ]

                    if (
                        context.output_text
                        and context.output_text != ""
                        and RegexGenerationRuleMatch.OUTPUT.value in rule["match"]
                    ):
                        match_against.append(html.unescape(context.output_text).strip())

                    if (
                        context.output_text
                        and context.output_text != ""
                        and RegexGenerationRuleMatch.OUTPUT_SENTENCE.value
                        in rule["match"]
                    ):
                        match_against += [
                            x.strip()
                            for x in re.split(
                                delimiters_regex_pattern, context.output_text
                            )
                            if x.strip() != ""
                        ]

                    if (
                        context.state
                        and "character_menu" in context.state
                        and context.state["character_menu"]
                        and context.state["character_menu"] != ""
                        and RegexGenerationRuleMatch.CHARACTER_NAME.value
                        in rule["match"]
                    ):
                        match_against.append(context.state["character_menu"])

                    if "negative_regex" in rule and any(
                        re.match(
                            normalize_regex(rule["negative_regex"]), x, re.IGNORECASE
                        )
                        for x in match_against
                    ):
                        continue

                    if "regex" in rule and not any(
                        re.match(normalize_regex(rule["regex"]), x, re.IGNORECASE)
                        for x in match_against
                    ):
                        continue

                if "actions" not in rule:
                    continue

                for action in rule["actions"]:
                    if action["name"] == "skip_generation":
                        return (
                            None,
                            "",
                            "",
                            context.params.base_prompt,
                            context.params.base_negative_prompt,
                        )

                    if action["name"] == "prompt_append" and "args" in action:
                        rules_prompt = _combine_prompts(rules_prompt, action["args"])

                    if action["name"] == "negative_prompt_append" "args" in action:
                        rules_negative_prompt += _combine_prompts(
                            rules_negative_prompt, action["args"]
                        )

                    if action["name"] == "faceswaplab_enable":
                        faceswaplab_force_enabled = True

                    if action["name"] == "faceswaplab_disable":
                        faceswaplab_force_enabled = False

                    if (
                        action["name"] == "faceswaplab_set_source_face"
                        and "args" in action
                    ):
                        faceswaplab_overwrite_source_face = action["args"]

                    if action["name"] == "reactor_enable":
                        reactor_force_enabled = True

                    if action["name"] == "reactor_disable":
                        reactor_force_enabled = False

                    if action["name"] == "reactor_set_source_face" and "args" in action:
                        reactor_overwrite_source_face = action["args"]

            except Exception as e:
                logger.error(
                    f"[SD WebUI Integration] Failed to apply rule: {rule['regex']}: {e}"
                )

    context_prompt = ""

    if context.params.trigger_mode == TriggerMode.INTERACTIVE and (
        context.params.interactive_mode_prompt_generation_mode
        == InteractiveModePromptGenerationMode.GENERATED_TEXT
        or InteractiveModePromptGenerationMode.DYNAMIC
    ):
        context_prompt = html.unescape(context.output_text or "")

    if context.params.trigger_mode == TriggerMode.CONTINUOUS and (
        context.params.continuous_mode_prompt_generation_mode
        == ContinuousModePromptGenerationMode.GENERATED_TEXT
    ):
        context_prompt = html.unescape(context.output_text or "")

    if ":" in context_prompt:
        context_prompt = (
            ", ".join(context_prompt.split(":")[1:])
            .replace(".", ",")
            .replace(":", ",")
            .strip()
            .strip("\n")
            .strip()
            .split("\n")[0]
            .strip()
            .lower()
        )

    generated_prompt = _combine_prompts(
        normalize_prompt(rules_prompt), normalize_prompt(context_prompt)
    )
    generated_negative_prompt = normalize_prompt(rules_negative_prompt)

    full_prompt = _combine_prompts(generated_prompt, context.params.base_prompt)

    full_negative_prompt = _combine_prompts(
        generated_negative_prompt, context.params.base_negative_prompt
    )

    logger.info
    (
        "[SD WebUI Integration] Using stable-diffusion-webui to generate images."
        + (
            f"\n"
            f"  Prompt: {full_prompt}\n"
            f"  Negative Prompt: {full_negative_prompt}"
        )
        if context.params.debug_mode_enabled
        else ""
    )

    try:
        response = sd_client.txt2img(
            prompt=full_prompt,
            negative_prompt=full_negative_prompt,
            seed=context.params.seed,
            sampler_name=context.params.sampler_name,
            enable_hr=context.params.upscaling_enabled,
            hr_scale=context.params.upscaling_scale,
            hr_upscaler=context.params.upscaling_upscaler,
            denoising_strength=context.params.denoising_strength,
            steps=context.params.sampling_steps,
            cfg_scale=context.params.cfg_scale,
            width=context.params.width,
            height=context.params.height,
            restore_faces=context.params.enhance_faces_enabled,
            override_settings_restore_afterwards=True,
            use_async=False,
        )

        response = cast(WebUIApiResult, response)

        if len(response.images) == 0:
            logger.error("[SD WebUI Integration] Failed to generate any images.")
            return (
                None,
                generated_prompt,
                generated_negative_prompt,
                full_prompt,
                full_negative_prompt,
            )

        formatted_result = ""
        style = 'style="width: 100%; max-height: 100vh;"'

        from ..script import EXTENSION_DIRECTORY_NAME

        for image in response.images:
            if faceswaplab_force_enabled or (
                faceswaplab_force_enabled is None and context.params.faceswaplab_enabled
            ):
                if context.params.debug_mode_enabled:
                    logger.info(
                        "[SD WebUI Integration] Using FaceSwapLab to swap faces."
                    )

                try:
                    response = sd_client.faceswaplab_swap_face(
                        image,
                        params=dataclasses.replace(
                            context.params,
                            faceswaplab_source_face=(
                                faceswaplab_overwrite_source_face
                                if faceswaplab_overwrite_source_face is not None
                                else context.params.faceswaplab_source_face
                            ).replace(
                                "{STABLE_DIFFUSION_EXTENSION_DIRECTORY}",
                                f"./extensions/{EXTENSION_DIRECTORY_NAME}",
                            ),
                        ),
                        use_async=False,
                    )
                    image = response.image
                except Exception as e:
                    logger.error(
                        f"[SD WebUI Integration] FaceSwapLab failed to swap faces: {e}"
                    )

            if reactor_force_enabled or (
                reactor_force_enabled is None and context.params.reactor_enabled
            ):
                if context.params.debug_mode_enabled:
                    logger.info("[SD WebUI Integration] Using ReActor to swap faces.")

                try:
                    response = sd_client.reactor_swap_face(
                        image,
                        params=dataclasses.replace(
                            context.params,
                            reactor_source_face=(
                                reactor_overwrite_source_face
                                if reactor_overwrite_source_face is not None
                                else context.params.reactor_source_face
                            ).replace(
                                "{STABLE_DIFFUSION_EXTENSION_DIRECTORY}",
                                f"./extensions/{EXTENSION_DIRECTORY_NAME}",
                            ),
                        ),
                        use_async=False,
                    )
                    image = response.image
                except Exception as e:
                    logger.error(
                        f"[SD WebUI Integration] ReActor failed to swap faces: {e}"
                    )

            if context.params.save_images:
                character = (
                    context.state.get("character_menu", "Default")
                    if context.state
                    else "Default"
                )

                file = f'{date.today().strftime("%Y_%m_%d")}/{character}_{int(time.time())}'  # noqa: E501

                # todo: do not hardcode extension path
                output_file = Path(
                    f"extensions/{EXTENSION_DIRECTORY_NAME}/outputs/{file}.png"
                )
                output_file.parent.mkdir(parents=True, exist_ok=True)

                image.save(output_file)
                image_source = f"/file/{output_file}"
            else:
                # resize image to avoid huge logs
                image.thumbnail((512, 512 * image.height / image.width))

                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                buffered.seek(0)
                image_bytes = buffered.getvalue()
                image_base64 = (
                    "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
                )
                image_source = image_base64

            formatted_result += f'<img src="{image_source}" {style}>\n'

    finally:
        attempt_vram_reallocation(VramReallocationTarget.LLM, context)

    return (
        formatted_result.rstrip("\n"),
        generated_prompt,
        generated_negative_prompt,
        full_prompt,
        full_negative_prompt,
    )


def _combine_prompts(prompt1: str, prompt2: str) -> str:
    if not prompt1 or prompt1 == "":
        return prompt2.strip(",").strip()

    if not prompt2 or prompt2 == "":
        return prompt1.strip(",").strip()

    return prompt1.strip(",").strip() + ", " + prompt2.strip(",").strip()
