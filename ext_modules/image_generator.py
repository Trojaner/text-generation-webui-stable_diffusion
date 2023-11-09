import base64
import io
import time
from datetime import date
from pathlib import Path
from typing import cast
from webuiapi import WebUIApiResult
from modules.logging_colors import logger
from ..context import GenerationContext
from ..params import TriggerMode
from .vram_manager import VramReallocationTarget, attempt_vram_reallocation


def normalize_prompt(prompt: str) -> str:
    if prompt is None or prompt.replace(",", "").replace("\n", "").strip() == "":
        return ""

    return prompt.strip().strip(",").strip().strip("\n")


def description_to_prompt(description: str) -> str:
    return normalize_prompt(
        description.replace("*", "")
        .replace('"', "")
        .replace(".", "")
        .replace("!", "")
        .replace("?", "")
        .replace("#", "")
        .replace(":", "")
        .replace(";", "")
        .split("\n")[0]
        .strip("(")
        .strip(")")
    )


def generate_html_images_for_context(context: GenerationContext) -> str | None:
    """
    Generates images for the given context using Stable Diffusion
    and returns the result as HTML output
    """

    attempt_vram_reallocation(VramReallocationTarget.STABLE_DIFFUSION, context)
    sd_client = context.sd_client

    if context.params.trigger_mode == TriggerMode.INTERACTIVE:
        full_prompt = (
            description_to_prompt(context.output_text) + ","
            if context.output_text
            else ""
        ) + normalize_prompt(context.params.base_prompt_suffix)
    else:
        full_prompt = (
            normalize_prompt(context.params.default_prompt)
            + ","
            + normalize_prompt(context.params.base_prompt_suffix)
        )

    full_negative_prompt = normalize_prompt(context.params.base_negative_prompt)

    logger.info(
        (
            "[SD WebUI Integration] Using stable-diffusion-webui to generate images.\n"
            f"  Prompt: {full_prompt}\n"
            f"  Negative Prompt: {full_negative_prompt}"
        )
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
            return None

        formatted_result = ""
        style = 'style="width: 100%; max-height: 100vh;"'
        for image in response.images:
            if context.params.faceswaplab_enabled:
                logger.info("[SD WebUI Integration] Using FaceSwapLab to swap faces.")

                try:
                    response = sd_client.faceswaplab_swap_face(
                        image,
                        face=context.params.faceswaplab_source_face,
                        use_async=False,
                    )
                    image = response.image
                except Exception as e:
                    logger.error(f"[SD WebUI Integration] Failed to swap faces: {e}")

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

    return formatted_result.rstrip("\n")
