from typing import Dict, List

import pytest
from transformers import AutoTokenizer

from vllm.sequence import Logprob, SamplingParams, Sequence, SequenceGroup
from vllm.transformers_utils.detokenizer import (Detokenizer,
                                                 detokenize_incrementally)
from vllm.transformers_utils.tokenizer_group import get_tokenizer_group

TRUTH = [
    "Hello here, this is a simple test",
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be used in production environments, where inference and serving",  # noqa
    "我很感谢你的热情"
]
TOKENIZERS = [
    "facebook/opt-125m",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "mosaicml/mpt-7b",
    "tiiuae/falcon-7b",
    "meta-llama/Llama-2-7b-hf",
    "codellama/CodeLlama-7b-hf",
]


def _run_incremental_decode(tokenizer, all_input_ids,
                            skip_special_tokens: bool, starting_index: int):
    decoded_text = ""
    offset = 0
    token_offset = 0
    prev_tokens = None
    for i in range(starting_index, len(all_input_ids)):
        new_tokens, text, offset, token_offset = detokenize_incrementally(
            tokenizer,
            all_input_ids[:i + 1],
            prev_tokens,
            offset,
            token_offset,
            skip_special_tokens=skip_special_tokens)
        decoded_text += text
        if prev_tokens is None:
            prev_tokens = new_tokens
        else:
            prev_tokens += new_tokens
    return decoded_text


@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("with_prompt", [True, False])
@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", (True, False))
def test_decode_streaming(tokenizer_id, truth, with_prompt,
                          skip_special_tokens):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if with_prompt:
        truth_tokens = tokenizer(truth, add_special_tokens=False)["input_ids"]
        prompt_input_ids = truth_tokens[:len(truth) // 2]
        generated_input_ids = truth_tokens[len(truth) // 2:]
        all_input_ids = prompt_input_ids + generated_input_ids
        starting_index = len(prompt_input_ids)
        prompt = tokenizer.decode(prompt_input_ids,
                                  skip_special_tokens=skip_special_tokens)
        generated = truth[len(prompt):]
    else:
        generated = truth
        starting_index = 0
        all_input_ids = tokenizer(truth, add_special_tokens=False)["input_ids"]
    if skip_special_tokens:
        if tokenizer.bos_token_id is not None:
            all_input_ids = [tokenizer.bos_token_id] + all_input_ids
            starting_index += 1
        all_input_ids = all_input_ids + [tokenizer.eos_token_id]

    decoded_text = _run_incremental_decode(
        tokenizer,
        all_input_ids,
        skip_special_tokens=skip_special_tokens,
        starting_index=starting_index)

    assert decoded_text == generated

    decoded_text = _run_incremental_decode(
        tokenizer, [len(tokenizer)],
        skip_special_tokens=skip_special_tokens,
        starting_index=starting_index)

    assert decoded_text == ''


@pytest.fixture
def detokenizer(tokenizer_name: str) -> Detokenizer:
    init_kwargs = dict(
        tokenizer_id=tokenizer_name,
        enable_lora=False,
        max_num_seqs=100,
        max_input_length=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        revision=None,
    )

    tokenizer_group = get_tokenizer_group(
        None,
        **init_kwargs,
    )

    return Detokenizer(tokenizer_group)


@pytest.fixture(name="complete_sequence_token_ids")
def create_complete_sequence_token_ids(complete_sequence: str,
                                       tokenizer_name: str) -> List[int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    complete_sequence_token_ids = tokenizer(complete_sequence)["input_ids"]
    return complete_sequence_token_ids


def create_sequence(prompt_token_ids=None):
    prompt_token_ids = prompt_token_ids or [1]
    return Sequence(
        seq_id=0,
        inputs={
            "prompt": "<s>",
            "prompt_token_ids": prompt_token_ids,
        },
        block_size=16,
    )


def create_dummy_logprobs(
        complete_sequence_token_ids: List[int]) -> List[Dict[int, Logprob]]:
    return [{
        token_id: Logprob(logprob=0.0),
        token_id + 1: Logprob(logprob=0.1)
    } for token_id in complete_sequence_token_ids]


def create_dummy_prompt_logprobs(
        complete_sequence_token_ids: List[int]) -> List[Dict[int, Logprob]]:
    # logprob for the first prompt token is not defined.
    return create_dummy_logprobs(complete_sequence_token_ids)[1:]


@pytest.mark.parametrize("complete_sequence", TRUTH)
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_decode_sequence_logprobs(complete_sequence: str,
                                  complete_sequence_token_ids: List[int],
                                  detokenizer: Detokenizer,
                                  skip_special_tokens: bool):
    """Verify Detokenizer decodes logprobs correctly."""
    sampling_params = SamplingParams(skip_special_tokens=skip_special_tokens,
                                     logprobs=2)

    # Run sequentially.
    seq = create_sequence()
    dummy_logprobs = create_dummy_logprobs(complete_sequence_token_ids)
    sequential_logprobs_text_chosen_token: List[str] = []
    sequential_logprobs_text_other_token: List[str] = []
    for new_token, logprobs in zip(complete_sequence_token_ids,
                                   dummy_logprobs):
        seq.append_token_id(new_token, logprobs)
        detokenizer.decode_sequence_inplace(seq, sampling_params)
        sequential_logprobs_text_chosen_token.append(
            seq.output_logprobs[-1][new_token].decoded_token)
        sequential_logprobs_text_other_token.append(
            seq.output_logprobs[-1][new_token + 1].decoded_token)
    sequential_result = seq.output_text

    assert sequential_result == "".join(sequential_logprobs_text_chosen_token)
    assert sequential_result != "".join(sequential_logprobs_text_other_token)

    if skip_special_tokens:
        # Text for logprobs for the chosen token should be the same as the
        # generated text. Note that this will only be true if we skip
        # special tokens.
        assert sequential_result == complete_sequence


@pytest.mark.parametrize("complete_sequence", TRUTH)
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", [True])
def test_decode_prompt_logprobs(complete_sequence: str,
                                complete_sequence_token_ids: List[int],
                                detokenizer: Detokenizer,
                                skip_special_tokens: bool):
    """Verify Detokenizer decodes prompt logprobs correctly."""
    sampling_params = SamplingParams(skip_special_tokens=skip_special_tokens,
                                     prompt_logprobs=1)

    # Run sequentially.
    seq = create_sequence(complete_sequence_token_ids)
    seq_group = SequenceGroup(request_id="1",
                              seqs=[seq],
                              sampling_params=sampling_params,
                              arrival_time=0.0)
    dummy_logprobs = create_dummy_prompt_logprobs(complete_sequence_token_ids)
    detokenizer.decode_prompt_logprobs_inplace(seq_group, dummy_logprobs)
    decoded_prompt_logprobs = dummy_logprobs

    if skip_special_tokens:
        # decoded_prompt_logprobs doesn't contain the first token.
        token_ids = complete_sequence_token_ids[1:]
        tokenzier = detokenizer.get_tokenizer_for_seq(seq)
        text = tokenzier.decode(token_ids,
                                skip_special_tokens=skip_special_tokens)
        # Text for logprobs for the chosen token should be the same as the
        # prompt text. Note that this will only be true if we skip
        # special tokens.
        assert text == "".join([
            logprobs[token_id].decoded_token
            for token_id, logprobs in zip(token_ids, decoded_prompt_logprobs)
        ])
        assert text != "".join([
            logprobs[token_id + 1].decoded_token
            for token_id, logprobs in zip(token_ids, decoded_prompt_logprobs)
        ])


@pytest.mark.parametrize("tokenizer_name", ["facebook/opt-125m"])
def test_decode_prompt_logprobs_pr_5846(detokenizer: Detokenizer):
    """ Regression test for PR #5846. """

    # This set of random input will generate incorrect output before #5846.
    prompt_token_ids = [3290, 1562, 8652, 3123, 1838, 9660]
    dummy_logprobs = [{
        1562: Logprob(logprob=0.0),
        3290: Logprob(logprob=0.1)
    }, {
        8652: Logprob(logprob=0.0),
        977: Logprob(logprob=0.1)
    }, {
        3123: Logprob(logprob=0.0),
        30: Logprob(logprob=0.1)
    }, {
        1838: Logprob(logprob=0.0),
        6: Logprob(logprob=0.1)
    }, {
        9660: Logprob(logprob=0.0),
        1316: Logprob(logprob=0.1)
    }]

    seq = create_sequence(prompt_token_ids)
    seq_group = SequenceGroup(
        request_id="1",
        seqs=[seq],
        sampling_params=SamplingParams(prompt_logprobs=1),
        arrival_time=0.0)

    detokenizer.decode_prompt_logprobs_inplace(seq_group, dummy_logprobs)
    decoded_prompt_logprobs = dummy_logprobs

    tokenzier = detokenizer.get_tokenizer_for_seq(seq)
    for logprobs in decoded_prompt_logprobs:
        for token_id, logprob in logprobs.items():
            assert tokenzier.decode(token_id) == logprob.decoded_token
