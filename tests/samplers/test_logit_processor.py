import random
from typing import Tuple
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.logit_processor import LogitProcessor
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner


class MockLogitsProcessor(LogitProcessor):

    def __init__(self, vocab_size: int, fake_logits: torch.Tensor):
        super().__init__(vocab_size=vocab_size)
        self.fake_logits = fake_logits

    def forward(self, *args, **kwargs):
        with patch(
                "vllm.model_executor.layers.logit_processor._prune_hidden_states",
                lambda x, y: x
        ), patch(
                "vllm.model_executor.layers.logit_processor.LogitProcessor._get_logits",
                lambda *args, **kwargs: self.fake_logits):
            return super().forward(*args, **kwargs)


def _prepare_test(
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsProcessor, ModelRunner]:
    vocab_size = 32000
    input_tensor = torch.rand((batch_size, 1024), dtype=torch.float16)
    fake_logits = torch.full((batch_size, vocab_size),
                             1e-2,
                             dtype=input_tensor.dtype)
    logit_processor = MockLogitsProcessor(32000, fake_logits)
    model_runner = ModelRunner(None, None, None, None, None)
    return input_tensor, fake_logits, logit_processor, model_runner


RANDOM_SEEDS = list(range(128))
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_logits_processors(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, logit_processor, model_runner = _prepare_test(
        batch_size)

    # This sample logits processor gives infinite score to the i-th token,
    # where i is the length of the input sequence.
    # We therefore expect the output token sequence to be [0, 1, 2, ...]
    def pick_ith(token_ids, logits):
        logits[len(token_ids)] = float("inf")
        return logits

    seq_group_metadata_list = []
    prompt_lens = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0,
                                               logits_processors=[pick_ith]),
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)
    logit_processor_output = logit_processor(
        embedding=None,
        hidden_states=input_tensor,
        sampling_metadata=sampling_metadata)
    assert logit_processor_output == fake_logits

    del model_runner
