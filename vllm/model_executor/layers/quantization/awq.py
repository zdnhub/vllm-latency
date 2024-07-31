from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  fused_experts_awq)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point})")

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    def get_quant_method(
            self, layer: torch.nn.Module,
            prefix: str) -> Optional[Union["AWQMoEMethod", "AWQLinearMethod"]]:
        if isinstance(layer, LinearBase):
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return AWQMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
                               pack_factor)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


class AWQMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # WEIGHTS
        w13_qweight = Parameter(torch.empty(num_experts,
                                            hidden_size,
                                            2 * intermediate_size //
                                            self.quant_config.pack_factor,
                                            dtype=torch.int32),
                                requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(
            w13_qweight, {
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
                "is_transposed": True,
                **extra_weight_attrs
            })

        w2_qweight = Parameter(torch.empty(num_experts,
                                           intermediate_size,
                                           hidden_size //
                                           self.quant_config.pack_factor,
                                           dtype=torch.int32),
                               requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(
            w2_qweight, {
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
                "is_transposed": True,
                **extra_weight_attrs
            })

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = Parameter(torch.empty(num_experts,
                                           hidden_size //
                                           self.quant_config.group_size,
                                           intermediate_size * 2,
                                           dtype=params_dtype),
                               requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, {
            "is_transposed": True,
            **extra_weight_attrs
        })

        w2_scales = Parameter(torch.empty(num_experts,
                                          intermediate_size //
                                          self.quant_config.group_size,
                                          hidden_size,
                                          dtype=params_dtype),
                              requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, {
            "is_transposed": True,
            **extra_weight_attrs
        })

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = Parameter(torch.empty(
            num_experts,
            hidden_size // self.quant_config.group_size,
            2 * intermediate_size // self.quant_config.pack_factor,
            dtype=torch.int32),
                               requires_grad=False)
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(
            w13_qzeros, {
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
                "is_transposed": True,
                **extra_weight_attrs
            })

        w2_qzeros = Parameter(torch.empty(
            num_experts,
            intermediate_size // self.quant_config.group_size,
            hidden_size // self.quant_config.pack_factor,
            dtype=torch.int32),
                              requires_grad=False)
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(
            w2_qzeros, {
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
                "is_transposed": True,
                **extra_weight_attrs
            })

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor) -> torch.Tensor:

        return fused_experts_awq(hidden_states=x,
                                 w1=layer.w13_qweight,
                                 w2=layer.w2_qweight,
                                 w1_scales=layer.w13_scales,
                                 w2_scales=layer.w2_scales,
                                 w1_qzeros=layer.w13_qzeros,
                                 w2_qzeros=layer.w2_qzeros,
                                 topk_weights=topk_weights,
                                 topk_ids=topk_ids,
                                 pack_factor=self.quant_config.pack_factor)
