import base64
from asyncio import Task
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List
from PIL import Image
from webuiapi import WebUIApi
from .params import FaceSwapLabParams, ReactorParams


@dataclass
class FaceSwapLabFaceSwapResponse:
    images: List[Image.Image]
    infos: List[str]

    @property
    def image(self) -> Image.Image:
        return self.images[0]


@dataclass
class ReactorFaceSwapResponse:
    image: Image.Image


class SdWebUIApi(WebUIApi):
    """
    This class extends the WebUIApi with some additional api endpoints.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def unload_checkpoint(self, use_async: bool = False) -> Task[None] | None:
        """
        Unload the current checkpoint from VRAM.
        """
        return self.post_and_get_api_result(  # type: ignore
            f"{self.baseurl}/unload-checkpoint", "", use_async
        )

    def reload_checkpoint(self, use_async: bool = False) -> Task[None] | None:
        """
        Reload the current checkpoint into VRAM.
        """

        return self.post_and_get_api_result(  # type: ignore
            f"{self.baseurl}/reload-checkpoint", "", use_async
        )

    def reactor_swap_face(
        self,
        target_image: Image.Image,
        params: ReactorParams,
        use_async: bool = False,
    ) -> Task[ReactorFaceSwapResponse] | ReactorFaceSwapResponse:
        """
        Swaps a face in an image using the ReActor extension.
        """
        buffer = BytesIO()
        target_image.save(buffer, format="PNG")
        target_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        source_image_base64 = None
        source_model = None

        reference_face_image_path = params.reactor_source_face
        reference_face_source = 0

        if reference_face_image_path.startswith("checkpoint://"):
            source_model = reference_face_image_path.replace("checkpoint://", "")
            reference_face_source = 1
        elif reference_face_image_path.startswith("data:image"):
            source_image_base64 = reference_face_image_path.split(",")[1]
        elif reference_face_image_path.startswith("file:///"):
            # todo: ensure path is inside text-generation-webui folder
            path = reference_face_image_path.replace("file:///", "")

            with open(path, "rb") as image_file:
                source_image_base64 = base64.b64encode(image_file.read()).decode()
        else:
            raise Exception(f"Failed to parse source face: {reference_face_image_path}")

        payload = {
            "source_image": source_image_base64 if reference_face_source == 0 else "",
            "target_image": target_image_base64,
            "source_faces_index": [params.reactor_source_face_index],
            "face_index": [params.reactor_target_face_index],
            "upscaler": params.reactor_upscaling_upscaler
            if params.reactor_upscaling_enabled
            else "None",
            "scale": params.reactor_upscaling_scale,
            "upscale_visibility": params.reactor_upscaling_visibility,
            "face_restorer": params.reactor_enhance_face_model
            if params.reactor_enhance_face_enabled
            else "None",
            "restorer_visibility": params.reactor_enhance_face_visibility,
            "codeformer_weight": params.reactor_enhance_face_codeformer_weight,
            "restore_first": 0 if params.reactor_enhance_face_upscale_first else 1,
            "model": params.reactor_model,
            "gender_source": params.reactor_source_gender,
            "gender_target": params.reactor_target_gender,
            "save_to_file": 0,
            "result_file_path": "",
            "device": params.reactor_device,
            "mask_face": 1 if params.reactor_mask_face else 0,
            "select_source": reference_face_source,
            "face_model": source_model if reference_face_source == 1 else "None",
            "source_folder": "",
        }

        return self.post_and_get_api_result(  # type: ignore
            f"{self.baseurl.replace('/sdapi/v1', '/reactor')}/image",
            payload,
            use_async,
        )

    def faceswaplab_swap_face(
        self,
        target_image: Image.Image,
        params: FaceSwapLabParams,
        use_async: bool = False,
    ) -> Task[FaceSwapLabFaceSwapResponse] | FaceSwapLabFaceSwapResponse:
        """
        Swaps a face in an image using the FaceSwapLab extension.
        """

        buffer = BytesIO()
        target_image.save(buffer, format="PNG")
        target_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        source_image_base64 = None
        source_face_checkpoint = None

        reference_face_image_path = params.faceswaplab_source_face

        if reference_face_image_path.startswith("checkpoint://"):
            source_face_checkpoint = reference_face_image_path.replace(
                "checkpoint://", ""
            )
        elif reference_face_image_path.startswith("data:image"):
            source_image_base64 = reference_face_image_path.split(",")[1]
        elif reference_face_image_path.startswith("file:///"):
            # todo: ensure path is inside text-generation-webui folder
            path = reference_face_image_path.replace("file:///", "")

            with open(path, "rb") as image_file:
                source_image_base64 = base64.b64encode(image_file.read()).decode()
        else:
            raise Exception(f"Failed to parse source face: {reference_face_image_path}")

        payload = {
            "image": target_image_base64,
            "units": [
                {
                    "source_img": source_image_base64,
                    "source_face": source_face_checkpoint,
                    "blend_faces": params.faceswaplab_blend_faces,
                    "same_gender": params.faceswaplab_same_gender_only,
                    "sort_by_size": params.faceswaplab_sort_by_size,
                    "check_similarity": False,
                    "compute_similarity": False,
                    "min_sim": 0,
                    "min_ref_sim": 0,
                    "faces_index": [params.faceswaplab_target_face_index],
                    "reference_face_index": params.faceswaplab_source_face_index,
                    "pre_inpainting": {
                        "inpainting_denoising_strengh": 0,
                        "inpainting_prompt": "Portrait of a [gender]",
                        "inpainting_negative_prompt": "blurry",
                        "inpainting_steps": 20,
                        "inpainting_sampler": "Default",
                        "inpainting_model": "Current",
                        "inpainting_seed": 0,
                    },
                    "swapping_options": {
                        "face_restorer_name": params.faceswaplab_enhance_face_model
                        if params.faceswaplab_enhance_face_enabled
                        else "None",
                        "restorer_visibility": params.faceswaplab_enhance_face_visibility,  # noqa: E501
                        "codeformer_weight": params.faceswaplab_enhance_face_codeformer_weight,  # noqa: E501
                        "upscaler_name": params.faceswaplab_upscaling_upscaler
                        if params.faceswaplab_upscaling_enabled
                        else "None",
                        "improved_mask": params.faceswaplab_mask_improved_mask_enabled,
                        "erosion_factor": params.faceswaplab_mask_erosion_factor,
                        "color_corrections": params.faceswaplab_color_corrections_enabled,  # noqa: E501
                        "sharpen": params.faceswaplab_sharpen_face,
                    },
                    "post_inpainting": {
                        "inpainting_denoising_strengh": 0,
                        "inpainting_prompt": "Portrait of a [gender]",
                        "inpainting_negative_prompt": "blurry",
                        "inpainting_steps": 20,
                        "inpainting_sampler": "Default",
                        "inpainting_model": "Current",
                        "inpainting_seed": 0,
                    },
                }
            ],
            "postprocessing": {
                "face_restorer_name": params.faceswaplab_postprocessing_enhance_face_model  # noqa: E501
                if params.faceswaplab_postprocessing_enhance_face_enabled
                else "None",
                "restorer_visibility": params.faceswaplab_postprocessing_enhance_face_visibility,  # noqa: E501
                "codeformer_weight": params.faceswaplab_postprocessing_enhance_face_codeformer_weight,  # noqa: E501
                "upscaler_name": params.faceswaplab_postprocessing_upscaling_upscaler
                if params.faceswaplab_postprocessing_upscaling_enabled
                else "None",
                "scale": params.faceswaplab_postprocessing_upscaling_scale,
                "upscaler_visibility": params.faceswaplab_postprocessing_upscaling_visibility,  # noqa: E501
                "inpainting_when": "After Upscaling/Before Restore Face",
                "inpainting_options": {
                    "inpainting_denoising_strengh": 0,
                    "inpainting_prompt": "Portrait of a [gender]",
                    "inpainting_negative_prompt": "blurry",
                    "inpainting_steps": 20,
                    "inpainting_sampler": "Default",
                    "inpainting_model": "Current",
                    "inpainting_seed": 0,
                },
            },
        }

        return self.post_and_get_api_result(  # type: ignore
            f"{self.baseurl.replace('/sdapi/v1', '/faceswaplab')}/swap_face",
            payload,
            use_async,
        )

    def refresh_vaes(self) -> Any:
        response = self.session.post(url=f"{self.baseurl}/refresh-vaes")
        return response.json()
