import base64
from io import BytesIO
from typing import Optional, Union

import aiohttp
from PIL import Image

from vllm.config import ModelConfig
from vllm.envs import VLLM_IMAGE_FETCH_TIMEOUT
from vllm.multimodal.image import ImagePixelData
from vllm.utils import make_async


class ImageFetchAiohttp:
    aiohttp_client: Optional[aiohttp.ClientSession] = None

    @classmethod
    def get_aiohttp_client(cls) -> aiohttp.ClientSession:
        if cls.aiohttp_client is None:
            timeout = aiohttp.ClientTimeout(total=VLLM_IMAGE_FETCH_TIMEOUT)
            connector = aiohttp.TCPConnector()
            cls.aiohttp_client = aiohttp.ClientSession(timeout=timeout,
                                                       connector=connector)

        return cls.aiohttp_client

    @classmethod
    async def close_aiohttp_client(cls) -> None:
        if cls.aiohttp_client:
            await cls.aiohttp_client.close()
            cls.aiohttp_client = None

    @classmethod
    async def fetch_image(cls, image_url: str) -> Image.Image:
        """Load image from a url or base64 encoded openai GPT4V format"""

        # Avoid circular import
        from vllm import __version__ as VLLM_VERSION

        client = cls.get_aiohttp_client()
        headers = {"User-Agent": f"vLLM/{VLLM_VERSION}"}

        if image_url.startswith('http'):
            async with client.get(url=image_url, headers=headers) as response:
                response.raise_for_status()
                # Open the image using PIL
                image = Image.open(BytesIO(response.content))

        elif image_url.startswith('data:image'):
            image = async_load_image_from_base64(image_url.split(',')[1])

        else:
            raise ValueError("Invalid image url: A valid image url must start "
                             "with either 'data:image' or 'http'.")

        return ImagePixelData(image)


def encode_image_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """encode image to base64 format."""

    buffered = BytesIO()
    if format == 'JPEG':
        image = image.convert('RGB')
    image.save(buffered, format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """Load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


async_load_image_from_base64 = make_async(load_image_from_base64)


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
