import asyncio
from vllm.utils import random_uuid
from io import StringIO
import argparse

import aiohttp

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionResponse,
                                              BatchRequestInput,
                                              BatchRequestOutput)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible batch runner.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help=
        "The path or url to a single input file. Currently supports local file "
        "paths, or the http protocol (http or https). If a URL is specified, "
        "the file should be available via HTTP GET.")
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=str,
        help=
        "The path or url to a single output file. Currently supports local file "
        "paths, or web (http or https) urls. If a URL is specified, the file"
        "should be available via HTTP PUT.")
    parser.add_argument("--response-role",
                        type=nullable_str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, "r") as f:
            return f.read()


async def write_file(path_or_url: str, data: str) -> None:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.put(path_or_url, data=data.encode("utf-8")):
            pass
    else:
        # We should make this async, but as long as this is always run as a
        # standalone program, blocking the event loop won't effect performance
        # in this particular case.
        with open(path_or_url, "w") as f:
            f.write(data)


async def run_request(chat_serving: OpenAIServingChat,
                      request: BatchRequestInput) -> BatchRequestOutput:
    chat_request = request.body
    chat_response = await chat_serving.create_chat_completion(chat_request)
    if isinstance(chat_response, ChatCompletionResponse):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=chat_response,
            error=None,
        )
    else:
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=None,
            error=chat_response,
        )
    return batch_output


async def main(args):
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)

    # When using single vLLM without engine_use_ray
    model_config = await engine.get_model_config()

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
    )

    response_futures = []
    for request_json in (await read_file(args.input_file)).strip().split("\n"):
        request = BatchRequestInput.model_validate_json(request_json)
        response_futures.append(run_request(openai_serving_chat, request))

    responses = await asyncio.gather(*response_futures)

    output_buffer = StringIO()
    for response in responses:
        print(response.model_dump_json(), file=output_buffer)

    output_buffer.seek(0)
    await write_file(args.output_file, output_buffer.read().strip())


if __name__ == "__main__":
    args = parse_args()

    logger.info("vLLM API server version %s", vllm.__version__)
    logger.info("args: %s", args)

    asyncio.run(main(args))
