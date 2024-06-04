import base64
from io import BytesIO
from typing import Union

import requests
from PIL import Image

from vllm.config import ModelConfig
from vllm.envs import VLLM_IMAGE_FETCH_TIMEOUT
from vllm.multimodal.image import ImagePixelData
from vllm.utils import make_async


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """Load image from base64 format"""
    return Image.open(BytesIO(base64.b64decode(image)))


def fetch_image(image_url: str) -> Image.Image:
    """Load image from a url or base64 encoded openai GPT4V format"""

    # Avoid circular import
    from vllm import __version__ as VLLM_VERSION

    headers = {"User-Agent": f"vLLM/{VLLM_VERSION}"}
    if image_url.startswith('http'):
        response = requests.get(image_url,
                                headers=headers,
                                timeout=VLLM_IMAGE_FETCH_TIMEOUT)
        response.raise_for_status()

        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
    elif image_url.startswith('data:image'):
        img = load_image_from_base64(image_url.split(',')[1])
    else:
        raise ValueError("Invalid image url: A valid image url must start "
                         "with either 'data:image' or 'http'.")

    return img


async_fetch_image = make_async(fetch_image)  # type: ignore


async def async_get_and_parse_image(image_url: str) -> ImagePixelData:
    with await async_fetch_image(image_url) as image:
        return ImagePixelData(image)


def get_full_image_text_prompt(image_prompt: str, text_prompt: str,
                               config: ModelConfig) -> str:
    """Combine image and text prompts for vision language model depending on
    the  model architecture."""

    if config.hf_config.model_type == "llava":
        full_prompt = f"{image_prompt}\n{text_prompt}"
    else:
        raise ValueError(
            f"Unsupported model type: {config.hf_config.model_type}")
    return full_prompt
