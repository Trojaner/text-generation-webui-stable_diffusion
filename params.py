import base64
from dataclasses import dataclass, field
from enum import Enum
import requests

default_description_prompt = """
You are now a text generator for the Stable Diffusion AI image generator. You will generate a text prompt for it.

Describe [subject] using comma-separated tags only. Do not use sentences.
Include many tags such as tags for the environment, gender, clothes, age, location, light, daytime, angle, pose, etc.

Do not write anything else. Do not ask any questions. Do not talk.
"""  # noqa E501


class TriggerMode(str, Enum):
    CONTINUOUS = "continuous"
    INTERACTIVE = "interactive"
    MANUAL = "manual"

    def __str__(self) -> str:
        return self.value


@dataclass
class ExtensionParams:
    display_name: str = field(default="Stable Diffusion")
    is_tab: bool = field(default=False)


@dataclass
class StableDiffusionClientParams:
    api_endpoint: str = field(default="http://127.0.0.1:7860/sdapi/v1")
    api_username: str | None = field(default=None)
    api_password: str | None = field(default=None)


@dataclass
class StableDiffusionGenerationParams:
    default_prompt: str = field(
        default="an adult female, close up, upper body, highlights in hair, brown eyes, wearing casual clothes, side light"  # noqa E501
    )
    base_prompt_suffix: str = field(
        default="high resolution, detailed, realistic, vivid"
    )
    base_negative_prompt: str = field(default="ugly, disformed, disfigured, immature")
    sampler_name: str = field(default="UniPC")
    denoising_strength: float = field(default=0.7)
    sampling_steps: int = field(default=25)
    width: int = field(default=512)
    height: int = field(default=512)
    cfg_scale: float = field(default=7)
    clip_skip: int = field(default=1)
    seed: int = field(default=-1)


@dataclass
class StableDiffusionPostProcessingParams:
    upscaling_enabled: bool = field(default=False)
    upscaling_upscaler: str = field(default="RealESRGAN 4x+")
    upscaling_scale: int = field(default=2)
    enhance_faces_enabled: bool = field(default=False)


@dataclass
class UserPreferencesParams:
    save_images: bool = field(default=True)
    trigger_mode: TriggerMode = field(default=TriggerMode.INTERACTIVE)
    interactive_mode_trigger_regex: str = field(
        default="(?aims)(send|mail|message|me)\\b.+?\\b(image|pic(ture)?|photo|snap(shot)?|selfie|meme)s?\\b"  # noqa E501
    )
    interactive_mode_subject_regex: str = field(default=".*\\s+of\\s+(.*)[\\.,!?]?")
    interactive_mode_description_prompt: str = field(default=default_description_prompt)
    interactive_mode_default_subject: str = field(
        default="your appearance, your surroundings and what you are doing right now"
    )
    dynamic_vram_reallocation_enabled: bool = field(default=False)
    dont_stream_when_generating_images: bool = field(default=True)


@dataclass
class FaceSwapLabParams:
    faceswaplab_enabled: bool = field(default=False)
    faceswaplab_same_gender_only: bool = field(default=True)
    faceswaplab_sort_by_size: bool = field(default=True)
    faceswaplab_source_face: str = field(
        default=("file:///extensions/stable_diffusion/assets/example_face.jpg")
    )
    faceswaplab_source_face_index: int = field(default=0)
    faceswaplab_enhance_face_enabled: bool = field(default=False)
    faceswaplab_enhance_face_model: str = field(default="CodeFormer")
    faceswaplab_enhance_face_visibility: int = field(default=1)
    faceswaplab_enhance_face_codeformer_weight: int = field(default=1)


@dataclass(kw_only=True)
class StableDiffusionWebUiExtensionParams(
    ExtensionParams,
    StableDiffusionClientParams,
    StableDiffusionGenerationParams,
    StableDiffusionPostProcessingParams,
    UserPreferencesParams,
    FaceSwapLabParams,
):
    def normalize(self) -> None:
        """
        Normalizes the parameters. This should be called after changing any parameters.
        """

        if self.api_username is not None and self.api_username.strip() == "":
            self.api_username = None

        if self.api_password is not None and self.api_password.strip() == "":
            self.api_password = None

        if self.faceswaplab_enabled and (
            self.faceswaplab_source_face.startswith("http://")
            or self.faceswaplab_source_face.startswith("https://")
        ):
            # todo: image may not be png format but for now it does not really matter
            self.faceswaplab_source_face = (
                "data:image/png;base64,"
                + base64.b64encode(
                    requests.get(self.faceswaplab_source_face).content
                ).decode()
            )
        pass
