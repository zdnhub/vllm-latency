import pytest

from vllm import LLM, SamplingParams


def test_multiple_sampling_params():

    llm = LLM(model="facebook/opt-125m",
              max_num_batched_tokens=4096,
              tensor_parallel_size=1)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(prompts) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(prompts, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(prompts, sampling_params=single_sampling_params)
    assert len(prompts) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(prompts, sampling_params=None)
    assert len(prompts) == len(outputs)


def test_chat():

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

    prompt1 = "Explain the concept of entropy."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt1
        },
    ]
    outputs = llm.chat(messages)
    assert len(outputs) == 1

    prompt2 = "Describe Bangkok in 150 words."
    messages = [messages] + [[
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt2
        },
    ]]
    outputs = llm.chat(messages)
    assert len(outputs) == len(messages)

    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)
    assert len(outputs) == len(messages)
