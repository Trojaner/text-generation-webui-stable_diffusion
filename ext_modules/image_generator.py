import base64
import io
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
    TriggerMode,
)
from .vram_manager import VramReallocationTarget, attempt_vram_reallocation


def normalize_prompt(prompt: str, do_additional_normalization: bool = False) -> str:
    if prompt is None:
        return ""

    result = (
        prompt.replace("*", "")
        .replace('"', "")
        .replace("!", ",")
        .replace("?", ",")
        .replace(";", ",")
        .strip()
        .strip("(")
        .strip(")")
        .strip(",")
        .strip()
    )

    if do_additional_normalization:
        result = (
            result.split("\n")[0]
            .replace(".", ",")
            .replace(":", "")
            .replace("*", "")
            .lower()
            .strip()
        )

    return result


def generate_html_images_for_context(
    context: GenerationContext,
) -> tuple[str | None, str, str, str]:
    """
    Generates images for the given context using Stable Diffusion
    and returns the result as HTML output
    """

    attempt_vram_reallocation(VramReallocationTarget.STABLE_DIFFUSION, context)
    sd_client = context.sd_client

    base_prompt = context.params.default_prompt
    do_additional_normalization = False

    if context.params.trigger_mode == TriggerMode.INTERACTIVE and (
        context.params.interactive_mode_prompt_generation_mode
        == InteractiveModePromptGenerationMode.GENERATED_TEXT
        or InteractiveModePromptGenerationMode.DYNAMIC
    ):
        base_prompt = context.output_text or ""
        do_additional_normalization = True

    if context.params.trigger_mode == TriggerMode.CONTINUOUS and (
        context.params.continuous_mode_prompt_generation_mode
        == ContinuousModePromptGenerationMode.GENERATED_TEXT
    ):
        base_prompt = context.output_text or ""
        do_additional_normalization = True

    base_prompt = normalize_prompt(
        base_prompt, do_additional_normalization=do_additional_normalization
    )

    full_prompt = (
        base_prompt
        + (", " if base_prompt and base_prompt != "" else "")
        + normalize_prompt(context.params.base_prompt_suffix)
    )

    full_negative_prompt = normalize_prompt(context.params.base_negative_prompt)

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
            return None, base_prompt, full_prompt, full_negative_prompt

        formatted_result = ""
        style = 'style="width: 100%; max-height: 100vh;"'
        for image in response.images:
            if context.params.faceswaplab_enabled:
                if context.params.debug_mode_enabled:
                    logger.info(
                        "[SD WebUI Integration] Using FaceSwapLab to swap faces."
                    )

                try:
                    response = sd_client.faceswaplab_swap_face(
                        image,
                        params=context.params,
                        use_async=False,
                    )
                    image = response.image
                except Exception as e:
                    logger.error(
                        f"[SD WebUI Integration] FaceSwapLab failed to swap faces: {e}"
                    )

            if context.params.reactor_enabled:
                if context.params.debug_mode_enabled:
                    logger.info("[SD WebUI Integration] Using Reactor to swap faces.")

                try:
                    response = sd_client.reactor_swap_face(
                        image,
                        params=context.params,
                        use_async=False,
                    )
                    image = response.image
                except Exception as e:
                    logger.error(
                        f"[SD WebUI Integration] Reactor failed to swap faces: {e}"
                    )

            if context.params.save_images:
                character = (
                    context.state.get("character_menu", "Default")
                    if context.state
                    else "Default"
                )

                file = f'{date.today().strftime("%Y_%m_%d")}/{character}_{int(time.time())}'  # noqa: E501

                # todo: do not hardcode extension path
                output_file = Path(f"extensions/stable_diffusion/outputs/{file}.png")
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

    return formatted_result.rstrip("\n"), base_prompt, full_prompt, full_negative_prompt
