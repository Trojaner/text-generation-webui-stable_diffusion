from typing import Any, List
import gradio as gr
from stringcase import sentencecase
from modules.logging_colors import logger
from modules.ui import refresh_symbol
from .context import GenerationContext
from .ext_modules.vram_manager import VramReallocationTarget, attempt_vram_reallocation
from .params import StableDiffusionWebUiExtensionParams as Params
from .params import TriggerMode
from .sd_client import SdWebUIApi

STATUS_SUCCESS = "#00FF00"
STATUS_PROGRESS = "#FFFF00"
STATUS_FAILURE = "#FF0000"

refresh_listeners: List[gr.components.Component] = []
connect_listeners: List[gr.components.Component] = []

status: gr.Label | None = None
status_text: str = ""

refresh_button: gr.Button | None = None

sd_client: SdWebUIApi | None = None
sd_samplers: List[str] = []
sd_upscalers: List[str] = []
sd_checkpoints: List[str] = []
sd_current_checkpoint: str = ""
sd_vaes: List[str] = []
sd_current_vae: str = ""

sd_connected: bool = False
sd_options: Any = None


def render_ui(params: Params) -> None:
    _render_status()
    _refresh_sd_data(params)

    _render_connection_details(params)
    _render_prompts(params)
    _render_models(params)
    _render_generation_parameters(params)

    with gr.Row():
        _render_chat_config(params)
        _render_faceswap_config(params)


def _render_connection_details(params: Params) -> None:
    global refresh_button

    with gr.Accordion("Connection details", open=True):
        with gr.Row():
            with gr.Column():
                api_username = gr.Textbox(
                    label="Username",
                    placeholder="Leave empty if no authentication is required",
                    value=lambda: params.api_username or "",
                )
                api_username.change(
                    lambda new_api_username: params.update(
                        Params(api_endpoint=new_api_username)
                    ),
                    api_username,
                    None,
                )

                api_password = gr.Textbox(
                    label="Password",
                    placeholder="Leave empty if no authentication is required",
                    value=lambda: params.api_password or "",
                    type="password",
                )
                api_password.change(
                    lambda new_api_password: params.update(
                        Params(api_endpoint=new_api_password)
                    ),
                    api_password,
                    None,
                )

            with gr.Column():
                api_endpoint = gr.Textbox(
                    label="API Endpoint",
                    placeholder=params.api_endpoint,
                    value=lambda: params.api_endpoint,
                )
                api_endpoint.change(
                    lambda new_api_endpoint: params.update(
                        Params(api_endpoint=new_api_endpoint)
                    ),
                    api_endpoint,
                    None,
                )

                refresh_button = gr.Button(
                    refresh_symbol + " Connect / refresh data",
                    interactive=True,
                )
                refresh_button.click(
                    lambda: _refresh_sd_data(params, force_refetch=True),
                    inputs=[],
                    outputs=refresh_listeners,
                )


def _render_prompts(params: Params) -> None:
    with gr.Accordion("Prompt Settings", open=True, visible=sd_connected) as prompts:
        connect_listeners.append(prompts)

        default_prompt = gr.Textbox(
            label="Default prompt",
            placeholder=params.default_prompt,
            value=lambda: params.default_prompt,
        )
        default_prompt.change(
            lambda new_prompt: params.update(Params(default_prompt=new_prompt)),
            default_prompt,
            None,
        )

        with gr.Row():
            base_prompt_suffix = gr.Textbox(
                label="Base prompt used for all prompts",
                placeholder=params.base_prompt_suffix,
                value=lambda: params.base_prompt_suffix,
            )
            base_prompt_suffix.change(
                lambda new_prompt: params.update(Params(base_prompt_suffix=new_prompt)),
                base_prompt_suffix,
                None,
            )

            base_negative_prompt = gr.Textbox(
                label="Base negative prompt used for all prompts",
                placeholder=params.base_negative_prompt,
                value=lambda: params.base_negative_prompt,
            )
            base_negative_prompt.change(
                lambda new_prompt: params.update(
                    Params(base_negative_prompt=new_prompt)
                ),
                base_negative_prompt,
                None,
            )


def _render_models(params: Params) -> None:
    with gr.Accordion("Models", open=True, visible=sd_connected) as models:
        connect_listeners.append(models)

        with gr.Row():
            global sd_current_checkpoint, sd_current_vae

            checkpoint = gr.Dropdown(
                label="Checkpoint",
                choices=sd_checkpoints,
                value=lambda: sd_current_checkpoint,  # checkpoint is not defined in params # noqa: E501
            )
            checkpoint.change(
                lambda new_checkpoint: _load_checkpoint(new_checkpoint, params),
                checkpoint,
                None,
            )
            refresh_listeners.append(checkpoint)

            vae = gr.Dropdown(
                label="VAE",
                choices=sd_vaes,
                value=lambda: sd_current_vae,  # vae is not defined in params
            )
            vae.change(
                lambda new_vae: _load_vae(new_vae, params),
                vae,
                None,
            )
            refresh_listeners.append(vae)


def _render_generation_parameters(params: Params) -> None:
    with gr.Accordion(
        "Generation Parameters", open=True, visible=sd_connected
    ) as generation_params:
        connect_listeners.append(generation_params)

        with gr.Row():
            with gr.Row("Image size"):
                width = gr.Number(
                    label="Width",
                    minimum=64,
                    maximum=2048,
                    value=lambda: params.width,
                )
                width.change(
                    lambda new_width: params.update(Params(width=new_width)),
                    width,
                    None,
                )

                height = gr.Number(
                    label="Height",
                    minimum=64,
                    maximum=2048,
                    value=lambda: params.height,
                )
                height.change(
                    lambda new_height: params.update(Params(height=new_height)),
                    height,
                    None,
                )

            with gr.Column():
                with gr.Row(elem_id="sampler_row"):
                    sampler_name = gr.Dropdown(
                        label="Sampling method",
                        choices=sd_samplers,
                        value=lambda: params.sampler_name,
                        elem_id="sampler_box",
                    )
                    sampler_name.change(
                        lambda new_sampler_name: params.update(
                            Params(sampler_name=new_sampler_name)
                        ),
                        sampler_name,
                        None,
                    )
                    refresh_listeners.append(sampler_name)

                    steps = gr.Slider(
                        label="Sampling steps",
                        minimum=1,
                        maximum=150,
                        value=lambda: params.sampling_steps,
                        step=1,
                        elem_id="steps_box",
                    )
                    steps.change(
                        lambda new_steps: params.update(
                            Params(sampling_steps=new_steps)
                        ),
                        steps,
                        None,
                    )

                    clip_skip = gr.Slider(
                        label="CLIP skip",
                        minimum=1,
                        maximum=4,
                        value=lambda: params.clip_skip,
                        step=1,
                        elem_id="clip_skip_box",
                    )
                    clip_skip.change(
                        lambda new_clip_skip: params.update(
                            Params(clip_skip=new_clip_skip)
                        ),
                        clip_skip,
                        None,
                    )

        with gr.Row():
            seed = gr.Number(
                label="Seed (use -1 for random)",
                value=lambda: params.seed,
                elem_id="seed_box",
            )
            seed.change(
                lambda new_seed: params.update(Params(seed=new_seed)), seed, None
            )

            cfg_scale = gr.Slider(
                label="CFG Scale",
                value=lambda: params.cfg_scale,
                minimum=1,
                maximum=30,
                elem_id="cfg_box",
            )
            cfg_scale.change(
                lambda new_cfg_scale: params.update(Params(cfg_scale=new_cfg_scale)),
                cfg_scale,
                None,
            )

            with gr.Column() as hr_options:
                restore_faces = gr.Checkbox(
                    label="Restore faces", value=lambda: params.enhance_faces_enabled
                )
                restore_faces.change(
                    lambda new_value: params.update(
                        Params(enhance_faces_enabled=new_value)
                    ),
                    restore_faces,
                    None,
                )

                enable_hr = gr.Checkbox(
                    label="Upscale image", value=lambda: params.upscaling_enabled
                )
                enable_hr.change(
                    lambda new_value: params.update(
                        Params(upscaling_enabled=new_value)
                    ),
                    enable_hr,
                    None,
                )

        with gr.Row(
            visible=params.upscaling_enabled, elem_classes="hires_opts"
        ) as hr_options:
            connect_listeners.append(hr_options)

            enable_hr.change(
                lambda enabled: hr_options.update(visible=enabled, inputs=hr_options),
                enable_hr,
                hr_options,
            )

            hr_upscaler = gr.Dropdown(
                label="Upscaler",
                choices=sd_upscalers,
                value=lambda: params.upscaling_upscaler,
                allow_custom_value=True,
            )
            hr_upscaler.change(
                lambda new_upscaler: params.update(
                    Params(upscaling_upscaler=new_upscaler)
                ),
                hr_upscaler,
                None,
            )
            refresh_listeners.append(hr_upscaler)

            hr_scale = gr.Slider(
                label="Upscale amount",
                minimum=1,
                maximum=4,
                value=lambda: params.upscaling_scale,
                step=0.1,
            )
            hr_scale.change(
                lambda new_value: params.update(Params(upscaling_scale=new_value)),
                hr_scale,
                None,
            )

            denoising_strength = gr.Slider(
                label="Denoising strength",
                minimum=0,
                maximum=1,
                value=lambda: params.denoising_strength,
                step=0.05,
            )
            denoising_strength.change(
                lambda new_value: params.update(Params(denoising_strength=new_value)),
                denoising_strength,
                None,
            )


def _render_faceswap_config(params: Params) -> None:
    with gr.Accordion("Faceswap", open=True, visible=sd_connected) as faceswap_config:
        connect_listeners.append(faceswap_config)

        with gr.Column():
            faceswap_enabled = gr.Checkbox(
                label="Enabled", value=lambda: params.faceswaplab_enabled
            )

            faceswap_enabled.change(
                lambda new_enabled: params.update(
                    Params(faceswaplab_enabled=new_enabled)
                ),
                faceswap_enabled,
                None,
            )

            faceswap_source_face = gr.Text(
                label="Source face",
                placeholder="See documentation for details...",
                value=lambda: params.faceswaplab_source_face,
            )

            faceswap_source_face.change(
                lambda new_source_face: params.update(
                    Params(faceswaplab_source_face=new_source_face)
                ),
                faceswap_source_face,
                None,
            )


def _render_chat_config(params: Params) -> None:
    with gr.Accordion("Chat Settings", open=True, visible=sd_connected) as chat_config:
        connect_listeners.append(chat_config)

        with gr.Row():
            mode = gr.Dropdown(
                label="Trigger mode",
                choices=[sentencecase(mode.value) for mode in TriggerMode],
                value=lambda: sentencecase(params.trigger_mode),
                type="index",
            )

            mode.change(
                lambda index: params.update(
                    Params(trigger_mode=TriggerMode.from_index(index))
                ),
                mode,
                None,
            )


def _render_status() -> None:
    global status
    status = gr.Label(lambda: status_text, label="Status", show_label=True)
    _set_status("Ready.", STATUS_SUCCESS)


def _refresh_sd_data(params: Params, force_refetch: bool = False) -> None:
    global sd_client, sd_connected, refresh_button

    if refresh_button is not None:
        refresh_button.update(interactive=False)

    sd_client = SdWebUIApi(
        baseurl=params.api_endpoint,
        username=params.api_username,
        password=params.api_password,
    )

    sd_connected = True
    _set_status("Connecting to Stable Diffusion WebUI...", STATUS_PROGRESS)

    if sd_connected and (force_refetch or sd_options is None):
        _fetch_sd_options(sd_client)

    if sd_connected and (force_refetch or len(sd_samplers) == 0):
        _fetch_samplers(sd_client)

    if sd_connected and (force_refetch or len(sd_upscalers) == 0):
        _fetch_upscalers(sd_client)

    if sd_connected and (force_refetch or len(sd_checkpoints) == 0):
        _fetch_checkpoints(sd_client)

    if sd_connected and (force_refetch or len(sd_vaes) == 0):
        _fetch_vaes(sd_client)

    for listener in connect_listeners:
        listener.update(visible=sd_connected)

    if refresh_button is not None:
        refresh_button.update(interactive=True)

    if not sd_connected:
        _set_status("Stable Diffusion WebUI connection failed", STATUS_FAILURE)
        return

    _set_status("✓ Connected to Stable Diffusion WebUI", STATUS_SUCCESS)


def _fetch_sd_options(sd_client: SdWebUIApi) -> None:
    _set_status("Fetching Stable Diffusion WebUI options...", STATUS_PROGRESS)

    global sd_options, sd_connected

    try:
        sd_options = sd_client.get_options()
    except BaseException as error:
        logger.error(error)
        sd_connected = False


def _fetch_samplers(sd_client: SdWebUIApi) -> None:
    _set_status("Fetching Stable Diffusion samplers...", STATUS_PROGRESS)

    global sd_samplers, sd_connected

    try:
        sd_samplers = [sampler["name"] for sampler in sd_client.get_samplers()]
    except BaseException as error:
        logger.error(error)
        sd_connected = False


def _fetch_upscalers(sd_client: SdWebUIApi) -> None:
    _set_status("Fetching Stable Diffusion upscalers...", STATUS_PROGRESS)

    global sd_upscalers, sd_connected

    try:
        sd_upscalers = [upscaler["name"] for upscaler in sd_client.get_upscalers()]
    except BaseException as error:
        logger.error(error)
        sd_connected = False


def _fetch_checkpoints(sd_client: SdWebUIApi) -> None:
    _set_status("Fetching Stable Diffusion checkpoints...", STATUS_PROGRESS)

    global sd_checkpoints, sd_current_checkpoint, sd_connected

    try:
        sd_client.refresh_checkpoints()

        sd_current_checkpoint = sd_options["sd_model_checkpoint"]
        sd_checkpoints = [
            checkpoint["title"] for checkpoint in sd_client.get_sd_models()
        ]
    except BaseException as error:
        logger.error(error)
        sd_connected = False


def _fetch_vaes(sd_client: SdWebUIApi) -> None:
    _set_status("Fetching Stable Diffusion VAEs...", STATUS_PROGRESS)

    global sd_vaes, sd_current_vae, sd_connected

    try:
        sd_client.refresh_vaes()
        sd_current_vae = sd_options["sd_vae"]
        sd_vaes = [checkpoint["model_name"] for checkpoint in sd_client.get_sd_vae()]
    except BaseException as error:
        logger.error(error)
        sd_connected = False


def _load_checkpoint(checkpoint: str, params: Params) -> None:
    global sd_client, sd_current_checkpoint
    sd_current_checkpoint = checkpoint

    assert sd_client is not None
    sd_client.set_options({"sd_model_checkpoint": checkpoint})

    # apply changes if dynamic VRAM allocation is not enabled
    # todo: check if model is loaded in VRAM via SD API instead of relying on vram reallocation check # noqa: E501
    if not params.dynamic_vram_reallocation_enabled:
        _set_status(
            f"Loading Stable Diffusion checkpoint: {checkpoint}...", STATUS_PROGRESS
        )
        sd_client.reload_checkpoint()

    _set_status("Reloading LLM model:...", STATUS_PROGRESS)

    attempt_vram_reallocation(
        VramReallocationTarget.LLM,
        GenerationContext(params=params, sd_client=sd_client),
    )

    _set_status(f"Stable Diffusion checkpoint ready: {checkpoint}.", STATUS_SUCCESS)


def _load_vae(vae: str, params: Params) -> None:
    global sd_client, sd_current_vae
    sd_current_vae = vae

    assert sd_client is not None
    sd_client.set_options({"sd_vae": vae})

    # apply changes if dynamic VRAM allocation is not enabled
    # todo: check if model is loaded in VRAM via SD API instead of relying on vram reallocation check # noqa: E501
    if not params.dynamic_vram_reallocation_enabled:
        _set_status(f"Loading Stable Diffusion VAE: {vae}...", STATUS_PROGRESS)
        sd_client.reload_checkpoint()

    attempt_vram_reallocation(
        VramReallocationTarget.LLM,
        GenerationContext(params=params, sd_client=sd_client),
    )

    _set_status(f"Stable Diffusion VAE ready: {vae}.", STATUS_SUCCESS)


def _set_status(text: str, status_color: str) -> None:
    global status, status_text
    assert status is not None

    status_text = text
    status.update(
        value=f'<span id="status-text" style="color: {status_color};">{status_text}</span>',  # noqa: E501
    )

    logger.info("[SD WebUI Integration] " + status_text)
