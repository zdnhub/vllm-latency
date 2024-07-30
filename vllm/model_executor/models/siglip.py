"""Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

import math
from typing import Iterable, List, Optional, Tuple

import torch
from einops import rearrange
from PIL import Image
from torch import nn
from transformers import SiglipConfig, SiglipVisionConfig
from vllm_flash_attn import flash_attn_varlen_func

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, ModelConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.inputs import LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.multimodal.image import (cached_get_tokenizer,
                                   repeat_and_pad_image_tokens)
from vllm.sequence import SequenceData
from vllm.utils import is_hip

from .utils import is_pp_missing_parameter


def get_siglip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size


def get_siglip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_siglip_patch_grid_length(image_size=image_size,
                                               patch_size=patch_size)
    return grid_length * grid_length


def get_siglip_image_feature_size(hf_config: SiglipVisionConfig) -> int:
    return get_siglip_num_patches(image_size=hf_config.image_size,
                                  patch_size=hf_config.patch_size)


def get_max_siglip_image_tokens(hf_config: SiglipVisionConfig) -> int:
    return get_siglip_image_feature_size(hf_config)


def dummy_seq_data_for_siglip(
    hf_config: SiglipVisionConfig,
    seq_len: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_siglip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    token_ids = [image_token_id] * image_feature_size
    token_ids += [0] * (seq_len - image_feature_size)
    return SequenceData(token_ids)


def dummy_image_for_siglip(
    hf_config: SiglipVisionConfig,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return {"image": image}


def input_processor_for_siglip(
    model_config: ModelConfig,
    hf_config: SiglipVisionConfig,
    llm_inputs: LLMInputs,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    if image_feature_size_override is None:
        image_feature_size = get_siglip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    new_prompt, new_token_ids = repeat_and_pad_image_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        image_token_id=image_token_id,
        repeat_count=image_feature_size,
    )

    # NOTE: Create a defensive copy of the original inputs
    return LLMInputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data=multi_modal_data,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/siglip/modeling_siglip.py#L249 # noqa
class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = VocabParallelEmbedding(
            self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions, dtype=torch.int64).expand(
                (1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int,
                                 width: int) -> torch.Tensor:
        """
        This method is an adapted method for SigLIP (due to SigLIP not having
        class embedding unlike other ViTs) that allows the model to interpolate
        the pre-trained position encodings such that it can be usable on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        position_embeddings = self.position_embedding.weight.unsqueeze(0)
        num_patches = embeddings.shape[1]
        num_positions = position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return position_embeddings

        dim = embeddings.shape[-1]
        height = height // self.patch_size
        width = width // self.patch_size
        # we add a small number to avoid floating point error
        # in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1

        patch_pos_embed = position_embeddings.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)),
            dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                height / math.sqrt(num_positions),
                width / math.sqrt(num_positions),
            ),
            mode="bicubic",
            align_corners=False,
        )
        if (int(height) != patch_pos_embed.shape[-2]
                or int(width) != patch_pos_embed.shape[-1]):
            raise ValueError("Width or height does not match with "
                             "the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self,
                pixel_values: torch.FloatTensor,
                interpolate_pos_encoding=False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(
                self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"Number of attention heads ({self.total_num_heads}) "
                "must be divisible by the tensor model parallel size"
                f" ({tp_size}).")

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.embed_dim // self.total_num_heads
        if self.head_dim * self.total_num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got "
                             "`embed_dim`: {self.embed_dim} and `num_heads`:"
                             f" {self.num_heads}).")
        self.qkv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
        )

        try:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scale,
                cache_config=cache_config,
                quant_config=quant_config,
            )
            self.attn_fn = self._vllm_attention_forward
            self.use_paged_attn = True
        except Exception:
            # For some pretrained Siglip models, the backend is not supported
            # (e.g. google/siglip-so400m-patch14-384 has hidden_size=1152
            #  with num_attention_heads=16, which is not supported)
            # If the backend is not supported, fall back to the default
            # TODO(ChristopherCho): flash_attn_varlen_func is not working properly
            # if self.qkv_proj.params_dtype in [torch.float16, torch.bfloat16]:
            #     # Flash attention only supports float16 and bfloat16
            #     self.attn_fn = self._flash_attention_forward
            #     self.use_paged_attn = False
            # else:
            #     self.attn_fn = self._basic_attention_forward
            #     self.use_paged_attn = False
            self.attn_fn = self._basic_attention_forward
            self.use_paged_attn = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        batch_size, q_len, _ = hidden_states.size()

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.split(
            [self.qkv_size] * 3, dim=-1)

        attn_output = self.attn_fn(
            q=query_states,
            k=key_states,
            v=value_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            batch_size=batch_size,
            q_len=q_len,
        )

        attn_output, _ = self.out_proj(attn_output)
        return attn_output

    def _vllm_attention_forward(self, q, k, v, kv_caches, attn_metadata, *args,
                                **kwargs):
        return self.attn(q, k, v, kv_caches, attn_metadata)

    def _basic_attention_forward(self, q, k, v, batch_size, q_len, *args,
                                 **kwargs):
        q = q.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)

        k_v_seq_len = k.shape[-2]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale

        if attn_weights.size() != (
                batch_size,
                self.num_heads,
                q_len,
                k_v_seq_len,
        ):
            raise ValueError(
                "Attention weights should be of size "
                f"{(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}")

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (
                batch_size,
                self.num_heads,
                q_len,
                self.head_dim,
        ):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        return attn_output

    def _flash_attention_forward(self, q, k, v, batch_size, q_len, *args,
                                 **kwargs):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the
                     query, key, and value. (B, S, H, D)
        """

        q = q.view(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, q_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, q_len, self.num_heads, self.head_dim)

        seqlen_k = k.shape[1]

        # goes for cuda device
        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=q.device,
        )

        # during training q,k,v always have same seqlen
        assert seqlen_k == q_len

        cu_seqlens_k = cu_seqlens_q
        dropout_p = self.dropout if self.training else 0.0

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            q_len,
            seqlen_k,
            dropout_p,
            softmax_scale=None,
            causal=False,
        )

        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        output = output.reshape(batch_size, q_len, self.embed_dim).contiguous()

        return output


class SiglipMLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)

        # For quantization, we require the hidden size to be a multiple of 64
        quantizable = (config.hidden_size % 64 == 0
                       and config.intermediate_size % 64 == 0)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config if quantizable else None,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config if quantizable else None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):

    def __init__(
        self,
        config: SiglipConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = SiglipAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            quant_config=quant_config,
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)`
                where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention
                layers. See `attentions` under returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        return outputs


class SiglipEncoder(nn.Module):

    def __init__(
        self,
        config: SiglipConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
            ) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> Tuple:
        last_hidden_state = inputs_embeds
        hidden_states = (last_hidden_state, )
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states=last_hidden_state,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )

            last_hidden_state = layer_outputs[0]
            hidden_states = hidden_states + (last_hidden_state, )

        return (last_hidden_state, hidden_states)


class SiglipVisionTransformer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.post_layernorm = nn.LayerNorm(embed_dim,
                                           eps=config.layer_norm_eps)
        self.use_head = (True if not hasattr(config, "vision_use_head") else
                         config.vision_use_head)
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(
                config=config, quant_config=quant_config)

    def forward(
        self,
        pixel_values,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        interpolate_pos_encoding: Optional[bool] = True,
    ) -> Tuple:
        r"""
        Returns:

        """
        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state) if self.use_head else None

        return (last_hidden_state, pooled_output) + encoder_outputs[1:]


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.head_size = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_size**-0.5
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config=config, quant_config=quant_config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionModel(nn.Module):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: SiglipVisionConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(
            config,
            cache_config,
            quant_config,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
            self,
            pixel_values,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: AttentionMetadata = None,
            interpolate_pos_encoding: Optional[bool] = False,  # added by eric
    ) -> Tuple:
        return self.vision_model(
            pixel_values=pixel_values,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path,
                tp_rank,
                tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type,
        ):
            if not isinstance(self.model.layers[layer_idx], nn.Identity):
                layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")
