from typing import (TYPE_CHECKING, List, Literal, Optional, Sequence,
                    TypedDict, Union, cast, overload)

from typing_extensions import NotRequired

from .utils import has_required_keys, is_dict, is_str

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


class ParsedText(TypedDict):
    content: str
    is_tokens: Literal[False]


class ParsedTokens(TypedDict):
    content: List[int]
    is_tokens: Literal[True]


# https://github.com/vllm-project/vllm/pull/4028
@overload
def parse_and_batch_prompt(
        prompt: Union[str, List[str]]) -> Sequence[ParsedText]:
    ...


@overload
def parse_and_batch_prompt(
        prompt: Union[List[int], List[List[int]]]) -> Sequence[ParsedTokens]:
    ...


def parse_and_batch_prompt(
    prompt: Union[str, List[str], List[int], List[List[int]]],
) -> Union[Sequence[ParsedText], Sequence[ParsedTokens]]:
    if isinstance(prompt, str):
        # case 1: a string
        return [ParsedText(content=prompt, is_tokens=False)]

    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")

        if isinstance(prompt[0], str):
            # case 2: array of strings
            return [
                ParsedText(content=elem, is_tokens=False)
                for elem in cast(List[str], prompt)
            ]
        if isinstance(prompt[0], int):
            # case 3: array of tokens
            elem = cast(List[int], prompt)
            return [ParsedTokens(content=elem, is_tokens=True)]
        if isinstance(prompt[0], list):
            if len(prompt[0]) == 0:
                raise ValueError("please provide at least one prompt")

            if isinstance(prompt[0][0], int):
                # case 4: array of token arrays
                return [
                    ParsedTokens(content=elem, is_tokens=True)
                    for elem in cast(List[List[int]], prompt)
                ]

    raise ValueError("prompt must be a string, array of strings, "
                     "array of tokens, or array of token arrays")


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


DecoderOnlyPromptInputs = Union[str, TextPrompt, TokensPrompt]
"""
Set of possible schemas for a single LLM input:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
"""


class ExplicitEncoderDecoderPrompt(TypedDict):
    """Represents an encoder/decoder model input prompt,
    comprising an encoder prompt and a decoder prompt.

    The encoder and decoder prompts, respectively,
    may formatted according to any of the
    DecoderOnlyPromptInputs schemas, and are not
    required to have the same schema.

    Only the encoder prompt may have multi-modal data.
    """

    encoder_prompt: DecoderOnlyPromptInputs

    decoder_prompt: DecoderOnlyPromptInputs


PromptInputs = Union[DecoderOnlyPromptInputs, ExplicitEncoderDecoderPrompt]
"""
Set of possible schemas for an LLM input, including
both decoder-only and encoder/decoder input types:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
- A single data structure containing both an encoder and a decoder prompt
  (:class:`ExplicitEncoderDecoderPrompt`)
"""


def get_prompt_type(prompt: Optional[PromptInputs], ) -> Optional[str]:
    """
    Get the type-name of the prompt argument instance, given that
    isinstance() cannot apply to TypedDict subclasses directly.
    If the prompt is None, return 'None' as the type name.

    Arguments:

    * prompt: LLM input prompt or None

    Returns:

    * String representation of prompt type
    """

    if prompt is None:
        return 'None'

    required_keys_dict = {
        'TextPrompt': ['prompt'],
        'TokensPrompt': ['prompt_token_ids'],
        'ExplicitEncoderDecoder': ['encoder_prompt', 'decoder_prompt'],
    }

    if is_dict(prompt):
        for ptype in required_keys_dict:
            # Ignore type checking in the conditional below because type
            # checker does not understand that is_dict(prompt) narrows
            # down the possible types
            if has_required_keys(
                    prompt,  # type: ignore
                    required_keys_dict[ptype]):
                return ptype

        raise ValueError(f"Invalid prompt {prompt}, valid types are "
                         "required_keys_dict={required_keys_dict}")
    elif is_str(prompt):
        return "str"

    raise ValueError(f"Invalid prompt {prompt}")


class LLMInputs(TypedDict):
    """
    The inputs in :class:`~vllm.LLMEngine` before they are
    passed to the model executor.
    """
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[Optional[str]]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    encoder_prompt_token_ids: NotRequired[List[int]]
    """The token IDs of the encoder prompt."""

    encoder_prompt: NotRequired[Optional[str]]
    """
    The original encoder prompt text corresponding to the token IDs, if
    available.
    """

    multi_modal_data: NotRequired[Optional["MultiModalDataDict"]]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """


def is_valid_encoder_decoder_llm_inputs(inputs: LLMInputs, ) -> bool:
    """
    Return True if the LLMInputs instance has the correct configuration
    for encoder/decoder.
    """

    if ('encoder_prompt_token_ids' in inputs
            and inputs['encoder_prompt_token_ids'] is not None):
        # Encoder prompt token ids field exists &
        # is not None
        return True

    return False
