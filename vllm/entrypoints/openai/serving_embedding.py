import time
from typing import AsyncIterator, List, Optional, Tuple

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.outputs import EmbeddingRequestOutput
from vllm.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)

TypeTokenIDs = List[int]


class OpenAIServingEmbedding(OpenAIServing):

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        served_model_names: List[str],
        *,
        log_requests: bool,
        max_log_len: Optional[int],
    ):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=None,
                         log_requests=log_requests,
                         max_log_len=max_log_len)
        self._check_embedding_mode(model_config.embedding_mode)

    async def create_embedding(self, request: EmbeddingRequest,
                               raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Return error for unsupported features.
        if request.encoding_format == "base64":
            return self.create_error_response(
                "base64 encoding is not currently supported")
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.monotonic())

        # Schedule the request and get the result generator.
        generators: List[AsyncIterator[EmbeddingRequestOutput]] = []
        try:
            pooling_params = request.to_pooling_params()

            prompts = list(
                self._tokenize_prompt_input_or_inputs(
                    request,
                    request.input,
                ))

            for i, prompt_inputs in enumerate(prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 prompt_inputs,
                                 pooling_params,
                                 lora_request=None)

                generator = self.engine.encode(
                    {"prompt_token_ids": prompt_inputs["prompt_token_ids"]},
                    pooling_params,
                    request_id_item,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, EmbeddingRequestOutput]] = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: List[Optional[EmbeddingRequestOutput]]
        final_res_batch = [None] * len(prompts)
        try:
            async for i, res in result_generator:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.engine.abort(f"{request_id}-{i}")
                    # TODO: Use a vllm-specific Validation Error
                    return self.create_error_response("Client disconnected")
                final_res_batch[i] = res

            final_res_batch_checked: List[EmbeddingRequestOutput] = []
            for final_res in final_res_batch:
                assert final_res is not None
                final_res_batch_checked.append(final_res)

            response = self.request_output_to_embedding_response(
                final_res_batch_checked,
                request_id,
                created_time,
                model_name,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def _check_embedding_mode(self, embedding_mode: bool):
        if not embedding_mode:
            logger.warning(
                "embedding_mode is False. Embedding API will not work.")
        else:
            logger.info("Activating the server engine with embedding enabled.")

    def request_output_to_embedding_response(
        self,
        final_res_batch: List[EmbeddingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> EmbeddingResponse:
        data = []
        num_prompt_tokens = 0
        for idx, final_res in enumerate(final_res_batch):
            assert final_res is not None
            prompt_token_ids = final_res.prompt_token_ids

            embedding_data = EmbeddingResponseData(
                index=idx, embedding=final_res.outputs.embedding)
            data.append(embedding_data)

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return EmbeddingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=data,
            usage=usage,
        )
