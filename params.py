import base64
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
import requests
from typing_extensions import Self

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

    @classmethod
    def index_of(cls, mode: Self) -> int:
        return list(TriggerMode).index(mode)

    @classmethod
    def from_index(cls, index: int) -> Self:
        return list(TriggerMode)[index]  # type: ignore

    def __str__(self) -> str:
        return self


class ContinuousModePromptGenerationMode(str, Enum):
    DEFAULT_PROMPT = "default_prompt"
    GENERATED_TEXT = "generated_text"

    @classmethod
    def index_of(cls, mode: Self) -> int:
        return list(ContinuousModePromptGenerationMode).index(mode)

    @classmethod
    def from_index(cls, index: int) -> Self:
        return list(ContinuousModePromptGenerationMode)[index]  # type: ignore

    def __str__(self) -> str:
        return self


class InteractiveModePromptGenerationMode(str, Enum):
    DEFAULT_PROMPT = "default_prompt"
    GENERATED_TEXT = "generated_text"
    DYNAMIC = "dynamic"

    @classmethod
    def index_of(cls, mode: Self) -> int:
        return list(InteractiveModePromptGenerationMode).index(mode)

    @classmethod
    def from_index(cls, index: int) -> Self:
        return list(InteractiveModePromptGenerationMode)[index]  # type: ignore

    def __str__(self) -> str:
        return self.value


class ReactorFace(int, Enum):
    NONE = 0
    FEMALE = 1
    MALE = 2

    @classmethod
    def index_of(cls, mode: Self) -> int:
        return list(ReactorFace).index(mode)

    @classmethod
    def from_index(cls, index: int) -> Self:
        return list(ReactorFace)[index]  # type: ignore


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
    upscaling_scale: float = field(default=2)
    enhance_faces_enabled: bool = field(default=False)


@dataclass
class UserPreferencesParams:
    save_images: bool = field(default=True)
    trigger_mode: TriggerMode = field(default=TriggerMode.INTERACTIVE)
    interactive_mode_input_trigger_regex: str = field(
        default="(?aims)(send|mail|message|me)\\b.+?\\b(image|pic(ture)?|photo|snap(shot)?|selfie|meme)s?\\b"  # noqa E501
    )
    interactive_mode_prompt_generation_mode: InteractiveModePromptGenerationMode = (
        field(default=InteractiveModePromptGenerationMode.DYNAMIC)
    )
    interactive_mode_subject_regex: str = field(default=".*\\s+of\\s+(.*)[\\.,!?]?")
    interactive_mode_description_prompt: str = field(default=default_description_prompt)
    interactive_mode_default_subject: str = field(
        default="your appearance, your surroundings and what you are doing right now"
    )
    continuous_mode_prompt_generation_mode: ContinuousModePromptGenerationMode = field(
        default=ContinuousModePromptGenerationMode.GENERATED_TEXT
    )
    dynamic_vram_reallocation_enabled: bool = field(default=False)
    dont_stream_when_generating_images: bool = field(default=True)


@dataclass
class FaceSwapLabParams:
    faceswaplab_enabled: bool = field(default=False)
    faceswaplab_source_face: str = field(
        default=("file:///extensions/stable_diffusion/assets/example_face.jpg")
    )
    faceswaplab_upscaling_enabled: bool = field(default=False)
    faceswaplab_upscaling_upscaler: str = field(default="RealESRGAN 4x+")
    faceswaplab_upscaling_scale: float = field(default=2)
    faceswaplab_upscaling_visibility: float = field(default=1)
    faceswaplab_postprocessing_upscaling_enabled: bool = field(default=False)
    faceswaplab_postprocessing_upscaling_upscaler: str = field(default="RealESRGAN 4x+")
    faceswaplab_postprocessing_upscaling_scale: float = field(default=2)
    faceswaplab_postprocessing_upscaling_visibility: float = field(default=1)
    faceswaplab_same_gender_only: bool = field(default=True)
    faceswaplab_sort_by_size: bool = field(default=True)
    faceswaplab_source_face_index: int = field(default=0)
    faceswaplab_target_face_index: int = field(default=0)
    faceswaplab_enhance_face_enabled: bool = field(default=False)
    faceswaplab_enhance_face_model: str = field(default="CodeFormer")
    faceswaplab_enhance_face_visibility: float = field(default=1)
    faceswaplab_enhance_face_codeformer_weight: float = field(default=1)
    faceswaplab_postprocessing_enhance_face_enabled: bool = field(default=False)
    faceswaplab_postprocessing_enhance_face_model: str = field(default="CodeFormer")
    faceswaplab_postprocessing_enhance_face_visibility: float = field(default=1)
    faceswaplab_postprocessing_enhance_face_codeformer_weight: float = field(default=1)
    faceswaplab_color_corrections_enabled: bool = field(default=False)
    faceswaplab_mask_erosion_factor: float = field(default=1)
    faceswaplab_mask_improved_mask_enabled: bool = field(default=False)
    faceswaplab_sharpen_face: bool = field(default=False)
    faceswaplab_blend_faces: bool = field(default=True)


@dataclass
class ReactorParams:
    reactor_enabled: bool = field(default=False)
    reactor_source_face: str = field(
        default=("file:///extensions/stable_diffusion/assets/example_face.jpg")
    )
    reactor_source_gender: ReactorFace = field(default=ReactorFace.NONE)
    reactor_target_gender: ReactorFace = field(default=ReactorFace.NONE)
    reactor_source_face_index: int = field(default=0)
    reactor_target_face_index: int = field(default=0)
    reactor_enhance_face_enabled: bool = field(default=False)
    reactor_enhance_face_model: str = field(default="CodeFormer")
    reactor_enhance_face_visibility: float = field(default=1)
    reactor_enhance_face_codeformer_weight: float = field(default=1)
    reactor_enhance_face_upscale_first: bool = field(default=False)
    reactor_upscaling_enabled: bool = field(default=False)
    reactor_upscaling_upscaler: str = field(default="RealESRGAN 4x+")
    reactor_upscaling_scale: float = field(default=2)
    reactor_upscaling_visibility: float = field(default=1)
    reactor_mask_face: bool = field(default=False)
    reactor_model: str = field(default="inswapper_128.onnx")
    reactor_device: str = field(default="CPU")


@dataclass(kw_only=True)
class StableDiffusionWebUiExtensionParams(
    StableDiffusionClientParams,
    StableDiffusionGenerationParams,
    StableDiffusionPostProcessingParams,
    UserPreferencesParams,
    FaceSwapLabParams,
    ReactorParams,
):
    display_name: str = field(default="Stable Diffusion")
    is_tab: bool = field(default=True)
    debug_mode_enabled: bool = field(default=False)

    def update(self, params: Self) -> None:
        """
        Updates the parameters.
        """

        for f in fields(self):
            val = getattr(params, f.name)

            if val == f.default or val == MISSING:
                continue

            if f.default_factory != MISSING:
                if val == f.default_factory():
                    continue

            setattr(self, f.name, val)

    def normalize(self) -> None:
        """
        Normalizes the parameters. This should be called after changing any parameters.
        """

        if self.api_username is not None and self.api_username.strip() == "":
            self.api_username = None

        if self.api_password is not None and self.api_password.strip() == "":
            self.api_password = None

        if isinstance(self.reactor_source_gender, str):
            self.reactor_source_gender = (
                ReactorFace[self.reactor_source_gender.upper()] or ReactorFace.NONE
            )

        if isinstance(self.reactor_target_gender, str):
            self.reactor_target_gender = (
                ReactorFace[self.reactor_target_gender.upper()] or ReactorFace.NONE
            )

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

        if self.reactor_enabled and (
            self.reactor_source_face.startswith("http://")
            or self.reactor_source_face.startswith("https://")
        ):
            # todo: same here issue as with faceswaplab above
            self.reactor_source_face = (
                "data:image/png;base64,"
                + base64.b64encode(
                    requests.get(self.reactor_source_face).content
                ).decode()
            )
