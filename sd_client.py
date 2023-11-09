import base64
from asyncio import Task
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List
from PIL import Image
from webuiapi import WebUIApi


@dataclass
class FaceSwapLabFaceSwapResponse:
    images: List[Image.Image]
    infos: List[str]

    @property
    def image(self) -> Image.Image:
        return self.images[0]


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

    def faceswaplab_swap_face(
        self, target_image: Image.Image, face: str, use_async: bool = False
    ) -> Task[FaceSwapLabFaceSwapResponse] | FaceSwapLabFaceSwapResponse:
        """
        Swap a face in an image using the FaceSwapLab extension.
        """

        buffer = BytesIO()
        target_image.save(buffer, format="PNG")
        target_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        source_img = None
        source_face = None

        if face.startswith("checkpoint://"):
            source_face = face.replace("checkpoint://", "")
        elif face.startswith("data:image"):
            source_img = face.split(",")[1]
        elif face.startswith("file:///"):
            # todo: ensure path is inside text-generation-webui folder
            path = face.replace("file:///", "")

            with open(path, "rb") as image_file:
                source_img = base64.b64encode(image_file.read()).decode()
        else:
            raise Exception(f"Failed to parse source face: {face}")

        payload = {
            "image": target_image_base64,
            "units": [
                {
                    "source_img": source_img,
                    "source_face": source_face,
                    "blend_faces": True,
                    "same_gender": False,
                    "sort_by_size": True,
                    "check_similarity": False,
                    "compute_similarity": False,
                    "min_sim": 0,
                    "min_ref_sim": 0,
                    "faces_index": [0],
                    "reference_face_index": 0,
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
                        "face_restorer_name": "CodeFormer",
                        "restorer_visibility": 1,
                        "codeformer_weight": 1,
                        "upscaler_name": "Lanczos",
                        "improved_mask": True,
                        "color_corrections": True,
                        "sharpen": False,
                        "erosion_factor": 1,
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
                "face_restorer_name": "CodeFormer",
                "restorer_visibility": 1,
                "codeformer_weight": 1,
                "upscaler_name": "None",
                "scale": 1,
                "upscaler_visibility": 1,
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
