"""1D GPT-2 model compatible with HuggingFace weights."""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import GPT2Config

from cacheflow.models import InputMetadata
from cacheflow.models.attention import GPTCacheFlowAttention
from cacheflow.models.sample import Sampler
from cacheflow.models.utils import (hf_model_weights_iterator,
                                    load_tensor_parallel_weights)
from cacheflow.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from cacheflow.parallel_utils.tensor_parallel import (VocabParallelEmbedding,
                                                      ColumnParallelLinear,
                                                      RowParallelLinear)
from cacheflow.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]


class GPT2Attention(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim ** -0.5

        self.c_attn = ColumnParallelLinear(self.hidden_size, 3 * self.hidden_size, bias=True,
                                           gather_output=False,
                                           perform_initialization=False)
        self.c_proj = RowParallelLinear(self.hidden_size, self.hidden_size, bias=True,
                                        input_is_parallel=True,
                                        perform_initialization=False)
        self.attn = GPTCacheFlowAttention(scale=self.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        key_cache, value_cache = kv_cache
        attn_output = self.attn(
            q, k, v, key_cache, value_cache, input_metadata, cache_event)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPT2MLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GPT2Config,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(hidden_size, intermediate_size,
                                         bias=True, gather_output=False,
                                         perform_initialization=False)
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size,
                                        bias=True, input_is_parallel=True,
                                        perform_initialization=False)

        assert config.activation_function == 'gelu_new'
        self.act = torch.nn.GELU(approximate='tanh')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        assert config.add_cross_attention == False
        assert config.scale_attn_by_inverse_layer_idx == False
        assert config.reorder_and_upcast_attn == False

        self.embed_dim = config.hidden_size
        assert config.vocab_size == 50257
        self.wte = VocabParallelEmbedding(50304, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                hidden_states, kv_caches[i], input_metadata, cache_event)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        # TODO(zhuohan): create a new weight after implementing pipeline
        #                parallelism
        self.lm_head_weight = self.transformer.wte.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.transformer(
            input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.lm_head_weight, hidden_states, input_metadata)
        return next_tokens

    _column_parallel_weights = ["wte.weight", "c_fc.weight", "c_fc.bias"]
    _row_parallel_weights = ["c_proj.weight"]

    def load_weights(self, model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache):
            if "lm_head.weight" in name:
                # GPT-2 ties the weights of the embedding layer and the final
                # linear layer.
                continue
            if ".attn.bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            name = "transformer." + name

            # Optimization: While the vocab size of GPT-2 is 50257, we
            # extend it to 50304 to make it divisible by 64.
            if name == "transformer.wte.weight":
                extra_rows = torch.empty(47, loaded_weight.shape[1]).to(
                    loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)

            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weight.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            param = state_dict[name]

            # For the fused QKV linear layer, manually shard the weights.
            if "c_attn" in name:
                # print(name, param.shape)
                shard_size = param.shape[0]
                # FIXME
                loaded_weight = loaded_weight[shard_size * tensor_model_parallel_rank:
                                              shard_size * (tensor_model_parallel_rank + 1)]
                # print(name, loaded_weight.shape)
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights)

    def initialize_dummy_weights(self) -> None:
        for param in self.state_dict().values():
            param.data.uniform_(-1e-3, 1e-3)
