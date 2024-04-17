import base64
from dataclasses import dataclass, field, fields
from enum import Enum
import requests
from typing_extensions import Self
from modules.logging_colors import logger

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


class IPAdapterAdapter(str, Enum):
    BASE = "Base"
    LIGHT = "Light"
    PLUS = "Plus"
    PLUS_FACE = "Plus Face"
    FULL_FACE = "Full face"
    BASE_SDXL = "Base SDXL"

    @classmethod
    def index_of(cls, mode: Self) -> int:
        return list(IPAdapterAdapter).index(mode)

    @classmethod
    def from_index(cls, index: int) -> Self:
        return list(IPAdapterAdapter)[index]  # type: ignore

    def __str__(self) -> str:
        return self


class ContinuousModePromptGenerationMode(str, Enum):
    STATIC = "static"
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
    STATIC = "static"
    GENERATED_TEXT = "generated_text"
    DYNAMIC = "dynamic"

    @classmethod
    def index_of(cls, mode: Self) -> int:
        return list(InteractiveModePromptGenerationMode).index(mode)

    @classmethod
    def from_index(cls, index: int) -> Self:
        return list(InteractiveModePromptGenerationMode)[index]  # type: ignore

    def __str__(self) -> str:
        return self


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
    base_prompt: str = field(
        default=(
            "RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, "
            "film grain, Fujifilm XT3"
        )
    )
    base_negative_prompt: str = field(
        default=(
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
            "sketch, cartoon, drawing, anime), text, cropped, out of frame, "
            "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, "
            "mutilated, extra fingers, mutated hands, poorly drawn hands, "
            "poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
            "bad proportions, extra limbs, cloned face, disfigured, gross proportions, "
            "malformed limbs, missing arms, missing legs, extra arms, extra legs, "
            "fused fingers, too many fingers, long neck"
        )
    )
    sampler_name: str = field(default="DPM SDE")
    sampling_steps: int = field(default=25)
    width: int = field(default=512)
    height: int = field(default=512)
    cfg_scale: float = field(default=6)
    clip_skip: int = field(default=1)
    seed: int = field(default=-1)


@dataclass
class StableDiffusionPostProcessingParams:
    upscaling_enabled: bool = field(default=False)
    upscaling_upscaler: str = field(default="RealESRGAN 4x+")
    upscaling_scale: float = field(default=2)
    hires_fix_enabled: bool = field(default=False)
    hires_fix_denoising_strength: float = field(default=0.2)
    hires_fix_sampler: str = field(default="UniPC")
    hires_fix_sampling_steps: int = field(default=10)
    restore_faces_enabled: bool = field(default=False)


@dataclass
class RegexGenerationRuleMatch(str, Enum):
    INPUT: str = "input"
    INPUT_SENTENCE: str = "input_sentence"
    OUTPUT: str = "output"
    OUTPUT_SENTENCE: str = "output_sentence"
    CHARACTER_NAME: str = "character_name"

    def __str__(self) -> str:
        return self


@dataclass
class RegexGenerationAction:
    name: str
    args: str | None


@dataclass
class RegexGenerationRule:
    regex: str | None
    negative_regex: str | None
    match: list[RegexGenerationRuleMatch] | None
    actions: list[RegexGenerationAction]


@dataclass
class UserPreferencesParams:
    save_images: bool = field(default=True)
    trigger_mode: TriggerMode = field(default=TriggerMode.INTERACTIVE)
    interactive_mode_input_trigger_regex: str = field(
        default=".*(send|upload|add|show|attach|generate)\\b.+?\\b(image|pic(ture)?|photo|snap(shot)?|selfie|meme)(s?)"  # noqa E501
    )
    interactive_mode_output_trigger_regex: str = field(
        default=".*[*([](sends|uploads|adds|shows|attaches|generates|here (is|are))\\b.+?\\b(image|pic(ture)?|photo|snap(shot)?|selfie|meme)(s?)"  # noqa E501
    )
    interactive_mode_prompt_generation_mode: InteractiveModePromptGenerationMode = (
        field(default=InteractiveModePromptGenerationMode.DYNAMIC)
    )
    interactive_mode_subject_regex: str = field(default=".*\\b(of)\\b(.+?)(?:[.!?]|$)")
    interactive_mode_description_prompt: str = field(default=default_description_prompt)
    interactive_mode_default_subject: str = field(
        default="your appearance, your surroundings and what you are doing right now"
    )
    continuous_mode_prompt_generation_mode: ContinuousModePromptGenerationMode = field(
        default=ContinuousModePromptGenerationMode.GENERATED_TEXT
    )
    dynamic_vram_reallocation_enabled: bool = field(default=False)
    dont_stream_when_generating_images: bool = field(default=True)
    generation_rules: dict | None = field(
        default=None
    )  # list[RegexGenerationRule] | None = field(default=None)


@dataclass
class FaceIDParams:
    faceid_enabled: bool = field(default=False)
    faceid_source_face: str = field(
        default=("file:///extensions/stable_diffusion/assets/example_face.jpg")
    )
    faceid_mode: list[str] = field(default_factory=lambda: ["FaceID", "FaceSwap"])
    faceid_model: str = field(default="FaceID Plus v2")
    faceid_override_sampler: bool = field(default=True)
    faceid_strength: float = field(default=1.0)
    faceid_structure: float = field(default=1.0)
    faceid_rank: int = field(default=128)
    faceid_tokens: int = field(default=4)
    faceid_cache_model: bool = field(default=False)


@dataclass
class IPAdapterParams:
    ipadapter_enabled: bool = field(default=False)
    ipadapter_adapter: IPAdapterAdapter = field(default=IPAdapterAdapter.BASE)
    ipadapter_reference_image: str = field(
        default=("file:///extensions/stable_diffusion/assets/example_face.jpg")
    )
    ipadapter_scale: float = field(default=0.5)


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
    faceswaplab_restore_face_enabled: bool = field(default=False)
    faceswaplab_restore_face_model: str = field(default="CodeFormer")
    faceswaplab_restore_face_visibility: float = field(default=1)
    faceswaplab_restore_face_codeformer_weight: float = field(default=1)
    faceswaplab_postprocessing_restore_face_enabled: bool = field(default=False)
    faceswaplab_postprocessing_restore_face_model: str = field(default="CodeFormer")
    faceswaplab_postprocessing_restore_face_visibility: float = field(default=1)
    faceswaplab_postprocessing_restore_face_codeformer_weight: float = field(default=1)
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
    reactor_restore_face_enabled: bool = field(default=False)
    reactor_restore_face_model: str = field(default="CodeFormer")
    reactor_restore_face_visibility: float = field(default=1)
    reactor_restore_face_codeformer_weight: float = field(default=1)
    reactor_restore_face_upscale_first: bool = field(default=False)
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
    FaceIDParams,
    IPAdapterParams,
):
    display_name: str = field(default="Stable Diffusion")
    is_tab: bool = field(default=True)
    debug_mode_enabled: bool = field(default=False)

    def update(self, params: dict) -> None:
        """
        Updates the parameters.
        """

        for f in params.keys():
            assert f in [x.name for x in fields(self)], f"Invalid field for params: {f}"

            val = params[f]
            setattr(self, f, val)

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

        # Todo: images are redownloaded and files are reread every time a text is generated. # noqa E501
        # This happens because normalize() is called on every generation and the downloaded values are not cached. # noqa E501

        if self.faceswaplab_enabled and (
            self.faceswaplab_source_face.startswith("http://")
            or self.faceswaplab_source_face.startswith("https://")
        ):
            try:
                self.faceswaplab_source_face = base64.b64encode(
                    requests.get(self.faceswaplab_source_face).content
                ).decode()
            except Exception as e:
                logger.exception(
                    "Failed to load FaceSwapLab source face image: %s", e, exc_info=True
                )
                self.faceswaplab_enabled = False

        if self.reactor_enabled and (
            self.reactor_source_face.startswith("http://")
            or self.reactor_source_face.startswith("https://")
        ):
            try:
                self.reactor_source_face = base64.b64encode(
                    requests.get(self.reactor_source_face).content
                ).decode()
            except Exception as e:
                logger.exception(
                    "Failed to load ReActor source face image: %s", e, exc_info=True
                )
                self.reactor_enabled = False

        if self.faceid_enabled:
            try:
                if self.faceid_source_face.startswith(
                    "http://"
                ) or self.faceid_source_face.startswith("https://"):
                    self.faceid_source_face = base64.b64encode(
                        requests.get(self.faceid_source_face).content
                    ).decode()

                if self.faceid_source_face.startswith("file:///"):
                    with open(
                        self.faceid_source_face.replace("file:///", ""), "rb"
                    ) as f:
                        self.faceid_source_face = base64.b64encode(f.read()).decode()
            except Exception as e:
                logger.exception(
                    "Failed to load FaceID source face image: %s", e, exc_info=True
                )
                self.faceid_enabled = False

        if self.ipadapter_enabled:
            try:
                if self.ipadapter_reference_image.startswith(
                    "http://"
                ) or self.ipadapter_reference_image.startswith("https://"):
                    self.ipadapter_reference_image = base64.b64encode(
                        requests.get(self.ipadapter_reference_image).content
                    ).decode()

                if self.ipadapter_reference_image.startswith("file:///"):
                    with open(
                        self.ipadapter_reference_image.replace("file:///", ""), "rb"
                    ) as f:
                        self.ipadapter_reference_image = base64.b64encode(
                            f.read()
                        ).decode()
            except Exception as e:
                logger.exception(
                    "Failed to load IP Adapter reference image: %s", e, exc_info=True
                )
                self.ipadapter_enabled = False
