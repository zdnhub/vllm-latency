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

    prompt = "Write a story about a bull rider named Tom who gets lost at sea the one time he tries to go fishing. Make it less than 300 words."
    messages = [
        {"role": "system", "content": "I am a helpful assistant"},
        {"role": "user", "content": prompt},
    ]
    outputs = llm.generate_chat(messages)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    prompt1 = "Write a story about a bull rider named Tom who gets lost at sea the one time he tries to go fishing. Make it less than 300 words."
    prompt2 = "Write a story about a pilot named Jane who finds a hidden treasure in the mountains. Make it less than 100 words."
    messages = [
        [
            {"role": "system", "content": "I am a helpful assistant"},
            {"role": "user", "content": prompt1},
        ],
        [
            {"role": "system", "content": "I am a helpful assistant"},
            {"role": "user", "content": prompt2},
        ]
    ]
    outputs = llm.generate_chat(messages)
    assert len(outputs) == 2
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
