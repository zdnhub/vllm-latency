import os

import pytest

from vllm import LLM, SamplingParams

oot_registration_file = os.path.join(os.path.dirname(__file__),
                                     "dummy_model.py")


# NOTE: order is important here, first test with tensor_parallel_size=2
# then test with tensor_parallel_size=1
# because CUDA_VISIBLE_DEVICES might be set in the first test
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_oot_registration(tensor_parallel_size):
    backend = os.environ.get("DISTRIBUTED_EXECUTOR_BACKEND", "mp")
    if backend == "ray" and tensor_parallel_size == 1:
        pytest.skip()
    prompts = ["Hello, my name is", "The text does not matter"]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(
        model="facebook/opt-125m",
        worker_init_callback_script=oot_registration_file,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=backend,
    )
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""
