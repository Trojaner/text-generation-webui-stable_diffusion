import base64
import dataclasses
import html
import io
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, cast
from partial_json_parser import loads
from PIL import Image
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


def normalize_prompt(prompt: str) -> str:
    if prompt is None:
        return ""

    result = (
        prompt.replace("*", "")
        .replace('"', "")
        .replace("!", ",")
        .replace("?", ",")
        .replace("&", "")
        .replace("\r", "")
        .replace("\n", ", ")
        .replace("*", "")
        .replace("#", "")
        .replace(".,", ",")
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
) -> tuple[str, str | None, str | None, str | None, str | None, str | None]:
    """
    Generates images for the given context using Stable Diffusion
    and returns the result as HTML output
    """

    attempt_vram_reallocation(VramReallocationTarget.STABLE_DIFFUSION, context)

    sd_client = context.sd_client

    output_text = context.output_text or ""

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
                        output_text
                        and output_text != ""
                        and RegexGenerationRuleMatch.OUTPUT.value in rule["match"]
                    ):
                        match_against.append(html.unescape(output_text).strip())

                    if (
                        output_text
                        and output_text != ""
                        and RegexGenerationRuleMatch.OUTPUT_SENTENCE.value
                        in rule["match"]
                    ):
                        match_against += [
                            x.strip()
                            for x in re.split(delimiters_regex_pattern, output_text)
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
                            output_text,
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
                    f"[SD WebUI Integration] Failed to apply rule: {rule['regex']}: %s",
                    e,
                    exc_info=True,
                )

    context_prompt = None

    if context.params.trigger_mode == TriggerMode.INTERACTIVE and (
        context.params.interactive_mode_prompt_generation_mode
        == InteractiveModePromptGenerationMode.GENERATED_TEXT
        or InteractiveModePromptGenerationMode.DYNAMIC
    ):
        context_prompt = html.unescape(output_text or "")

    if context.params.trigger_mode == TriggerMode.CONTINUOUS and (
        context.params.continuous_mode_prompt_generation_mode
        == ContinuousModePromptGenerationMode.GENERATED_TEXT
    ):
        context_prompt = html.unescape(output_text or "")

    if context.params.trigger_mode == TriggerMode.TOOL:
        output_text = html.unescape(output_text or "").strip()

        json_search = re.search(
            r"(\b)?([{\[].*[\]}])(\b)?", output_text, flags=re.I | re.M | re.S | re.U
        )

        if not json_search:
            logger.warning(
                "No JSON output found in the output text: %s.\nTry enabling JSON grammar rules to avoid such errors.",
                output_text,
            )

        json_text_original = json_search.group(0) if json_search else "{}"

        try:
            json_text = (
                json_text_original.strip()
                .replace("\r\n", "\n")
                .replace("'", "")
                .replace("“", '"')  # yes, this actually happened.
                .replace("”", '"')  # llms are really creative and crazy...
                .replace(
                    "{{", "{ {"
                )  # for some reason the json parser doesnt like this
                .replace("}}", "} }")
            )
        except Exception as e:
            logger.warning(
                "JSON extraction from text failed: %s\n%s.\n\nTry enabling JSON grammar rules to avoid such errors.",
                repr(e),
                output_text,
            )

            json_text = "{}"

        output_text = (
            output_text.replace(json_text_original + "\n", "")
            .replace("\n" + json_text_original, "")
            .replace(json_text_original, "")
            .replace("Action: ```json\n", "")
            .replace("Action: ```json", "")
            .replace("Action:\n", "")
            .replace("Action:", "")
            .replace("\n```json", "")
            .replace("```json", "")
            .replace("```json\n", "")
            .replace("\n```", "")
            .replace("```", "")
            .strip("\r\n")
            .strip("\n")
            .strip()
        )

        json = None

        if json_search and json_text and json_text not in ["[]", "{}", "()"]:
            try:
                json = loads(json_text)
            except Exception as e:
                logger.warning(
                    "Failed to parse JSON from output text: %s\n%s\n\nTry enabling JSON grammar rules to avoid such errors.",
                    repr(e),
                    json_text,
                    exc_info=True,
                )

        if json is not None:
            tools: list[Any] = json if isinstance(json, list) else [json]

            for tool in tools:
                tool_name: str = (
                    tool.get("tool", None)
                    or tool.get("tool name", None)
                    or tool.get("tool_name", None)
                    or tool.get("tool call", None)
                    or tool.get("tool_call", None)
                    or tool.get("name", None)
                    or tool.get("function", None)
                    or tool.get("function_name", None)
                    or tool.get("function name", None)
                    or tool.get("function_call", None)
                    or tool.get("function call", None)
                )

                tool_params: dict = (
                    tool.get("tool_parameters", None)
                    or tool.get("tool parameters", None)
                    or tool.get("parameters", None)
                    or tool.get("tool_params", None)
                    or tool.get("tool params", None)
                    or tool.get("params", None)
                    or tool.get("tool_arguments", None)
                    or tool.get("tool arguments", None)
                    or tool.get("arguments", None)
                    or tool.get("tool_args", None)
                    or tool.get("tool args", None)
                    or tool.get("args", None)
                )

                if not tool_name or not tool_params:
                    continue

                if tool_name.lower() in [
                    "generate_image",
                    "generate image",
                    "generateimage",
                ]:
                    context_prompt = (
                        tool_params.get("text", None)
                        or tool_params.get("prompt", None)
                        or tool_params.get("query", None)
                        or ""
                    )

                if tool_name.lower() in ["add_text", "add text", "addtext"]:
                    tool_text = (
                        tool_params.get("text", None)
                        or tool_params.get("prompt", None)
                        or tool_params.get("query", None)
                        or ""
                    )
                    output_text = tool_text + (
                        "\n" + output_text if output_text else ""
                    )

    if context_prompt is None:
        return (
            output_text,
            None,
            None,
            None,
            None,
            None,
        )

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

    generated_prompt = _combine_prompts(rules_prompt, normalize_prompt(context_prompt))
    generated_negative_prompt = rules_negative_prompt

    full_prompt = _combine_prompts(generated_prompt, context.params.base_prompt)

    full_negative_prompt = _combine_prompts(
        generated_negative_prompt, context.params.base_negative_prompt
    )

    debug_info = (
        (
            f"\n"
            f"  Prompt: {full_prompt}\n"
            f"  Negative Prompt: {full_negative_prompt}"
        )
        if context.params.debug_mode_enabled
        else ""
    )

    logger.info(
        "[SD WebUI Integration] Using stable-diffusion-webui to generate images. %s",
        debug_info,
    )

    try:
        response = sd_client.txt2img(
            prompt=full_prompt,
            negative_prompt=full_negative_prompt,
            seed=context.params.seed,
            sampler_name=context.params.sampler_name,
            full_quality=True,
            enable_hr=context.params.upscaling_enabled
            or context.params.hires_fix_enabled,
            hr_scale=context.params.upscaling_scale,
            hr_upscaler=context.params.upscaling_upscaler,
            denoising_strength=context.params.hires_fix_denoising_strength,
            hr_sampler=context.params.hires_fix_sampler,
            hr_force=context.params.hires_fix_enabled,
            hr_second_pass_steps=(
                context.params.hires_fix_sampling_steps
                if context.params.hires_fix_enabled
                else 0
            ),
            steps=context.params.sampling_steps,
            cfg_scale=context.params.cfg_scale,
            width=context.params.width,
            height=context.params.height,
            restore_faces=context.params.restore_faces_enabled,
            faceid_enabled=context.params.faceid_enabled,
            faceid_mode=context.params.faceid_mode,
            faceid_model=context.params.faceid_model,
            faceid_image=context.params.faceid_source_face,
            faceid_scale=context.params.faceid_strength,
            faceid_structure=context.params.faceid_structure,
            faceid_rank=context.params.faceid_rank,
            faceid_override_sampler=context.params.faceid_override_sampler,
            faceid_tokens=context.params.faceid_tokens,
            faceid_cache_model=context.params.faceid_cache_model,
            ipadapter_enabled=context.params.ipadapter_enabled,
            ipadapter_adapter=context.params.ipadapter_adapter,
            ipadapter_scale=context.params.ipadapter_scale,
            ipadapter_image=context.params.ipadapter_reference_image,
            use_async=False,
        )

        response = cast(WebUIApiResult, response)

        if len(response.images) == 0:
            logger.error("[SD WebUI Integration] Failed to generate any images.")
            return (
                output_text,
                None,
                generated_prompt,
                generated_negative_prompt,
                full_prompt,
                full_negative_prompt,
            )

        formatted_result = ""
        style = 'style="width: 100%; max-height: 100vh;"'

        from ..script import EXTENSION_DIRECTORY_NAME

        image: Image.Image
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
                    image = response.image  # type: ignore
                except Exception as e:
                    logger.error(
                        "[SD WebUI Integration] FaceSwapLab failed to swap faces: %s",
                        e,
                        exc_info=True,
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
                    image = response.image  # type: ignore
                except Exception as e:
                    logger.error(
                        "[SD WebUI Integration] ReActor failed to swap faces: %s",
                        e,
                        exc_info=True,
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
                image.thumbnail((512, int(512 * image.height / image.width)))

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
        output_text,
        formatted_result.rstrip("\n"),
        generated_prompt,
        generated_negative_prompt,
        full_prompt,
        full_negative_prompt,
    )


def _combine_prompts(prompt1: str, prompt2: str) -> str:
    if prompt1 is None and prompt2 is None:
        return ""

    if prompt1 is None or prompt1 == "":
        return prompt2.strip(",").strip()

    if prompt2 is None or prompt2 == "":
        return prompt1.strip(",").strip()

    return prompt1.strip(",").strip() + ", " + prompt2.strip(",").strip()
