from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

try:
    import bitblas
    from bitblas.utils import auto_detect_nvidia_target
except ImportError as e:
    bitblas_import_exception = e
    raise ValueError(
        "Trying to use the bitblas backend, but could not import dependencies"
        f"with the following error: {bitblas_import_exception}"
    ) from bitblas_import_exception

BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = bitblas.cache.get_database_path()

BITBLAS_SUPPORTED_NUM_BITS = [1, 2, 4, 8]
BITBLAS_SUPPORTED_SYM = [False, True]


class BitBLASConfig(QuantizationConfig):
    """Config class for BitBLAS.

    Reference: https://github.com/Microsoft/BitBLAS
    """
    TORCH_DTYPE = torch.float16
    STORAGE_DTYPE = "int8"  # assume int8 storage
    TORCH_STORAGE_DTYPE = getattr(torch, STORAGE_DTYPE)
    # "original" or "rescale" or "quantized",
    # gptq_with_bitblas prefer "quantized implementation"
    ZEROS_MODE = "quantized"

    def __init__(self, weight_bits: int, group_size: Optional[int],
                 desc_act: Optional[bool], is_sym: Optional[bool],
                 quant_method: Optional[str]) -> None:
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym
        self.quant_method = quant_method

        # Verify
        if self.weight_bits not in BITBLAS_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"BitBLAS does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {BITBLAS_SUPPORTED_NUM_BITS} "
                "are supported.")

        if self.is_sym not in BITBLAS_SUPPORTED_SYM:
            raise ValueError(
                f"BitBLAS does not support is_sym = {self.is_sym}. "
                f"Only sym = {BITBLAS_SUPPORTED_SYM} are supported.")

        storage_dtype = self.STORAGE_DTYPE
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))

        self.storage_dtype = storage_dtype
        self.storage_torch_dtype = self.TORCH_STORAGE_DTYPE
        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = storage_nbit // weight_bits
        self.nbits = weight_bits

        # Zeros type for the quantized weights.
        self.zeros_mode = self.ZEROS_MODE
        # set input bits if bitnet
        self.input_bits: Optional[int] = None
        if self.quant_method == "bitnet":
            self.input_bits = 8

    def __repr__(self) -> str:
        return (f"BitBLASConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"is_sym={self.is_sym}, "
                f"quant_method={self.quant_method})")

    @classmethod
    def get_name(cls) -> str:
        return "bitblas"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @staticmethod
    def get_from_keys(config: Dict[str, Any],
                      keys: List[str],
                      default: Any = None) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        return default

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitBLASConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"], -1)
        desc_act = cls.get_from_keys(config, ["desc_act"], False)
        is_sym = cls.get_from_keys(config, ["sym"], False)
        quant_method = cls.get_from_keys(config, ["quant_method"])
        return cls(weight_bits, group_size, desc_act, is_sym, quant_method)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        # compat: autogptq >=0.8.0 use checkpoint_format: str
        # compat: autogptq <=0.7.1 is_bitblas_format: bool
        is_bitblas_format = (hf_quant_cfg.get("checkpoint_format") == "bitblas"
                             or hf_quant_cfg.get("is_bitblas_format", False))

        is_valid_user_quant = (user_quant is None or user_quant == "gptq"
                               or user_quant == "bitblas")

        if is_bitblas_format and is_valid_user_quant:
            msg = ("The model is serialized in {} format. Using {} kernel.".
                   format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        return None

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["BitBLASLinearMethod"]:
        if isinstance(layer, LinearBase):
            return BitBLASLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class BitBLASLinearMethod(LinearMethodBase):
    """Linear method for BitBLAS.

    Args:
        quant_config: The BitBLAS quantization config.
    """
    OPT_FEATURES = [1, 16, 32, 64, 128, 256, 512, 1024]
    ENABLE_TUNING = True
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    def __init__(self, quant_config: BitBLASConfig):
        self.quant_config = quant_config
        if self.quant_config.quant_method == "bitnet":
            input_bits = self.quant_config.input_bits
            if input_bits is None:
                raise ValueError("input_bits must be set for bitnet")
            self.Qp = 2**(input_bits - 1) - 1
            self.Qn = -2**(input_bits - 1)
        else:
            self.Qp = None
            self.Qn = None

    def create_weights_bitnet(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Creates quantized weights for use in linear operations.

        The function initializes and returns a dictionary containing quantized 
        weights, scales, and zeros
        for performing quantized matrix multiplication operations.

        Args:
            input_size_per_partition: The size of the input partition.
            output_size_per_partition: The size of the output partition.
            input_size: The total size of the input (unused).
            output_size: The total size of the output (unused).
            params_dtype: 
                The data type of the parameters (expected to be torch.float16).

        Returns:
            A dictionary containing the quantized weights ('qweight'), 
            scales ('scales'), and zeros ('zeros').

        Raises:
            ValueError: If `params_dtype` is not `torch.float16` or if the 
            input size per partition is not divisible by the group size in 
            `quant_config`.
        """
        del input_size, output_size  # Unused arguments.
        if params_dtype != torch.float16:
            raise ValueError("Parameter data type must be torch.float16, "
                             f"but got {params_dtype}")

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)

        # Initialize or retrieve the BitBLAS matrix multiplication operator.
        self._configure_bitblas_matmul(input_size_per_partition,
                                       output_size_per_partition,
                                       params_dtype=torch.int8,
                                       enable_tuning=self.ENABLE_TUNING,
                                       bias=False,
                                       layout="nt",
                                       bits=self.quant_config.weight_bits,
                                       out_dtype="float32")

        # Initialize quantized weights with dimensions

        qweight = Parameter(
            torch.empty(
                self.bitblas_matmul.retrieve_weight_shape(),
                device="cuda",
                dtype=self.quant_config.storage_torch_dtype,
            ),
            requires_grad=False,
        )
        # Attributes to help with unpacking and applying the weights later.
        set_weight_attrs(
            qweight,
            {
                "input_dim":
                1,
                "output_dim":
                0,
                "packed_dim":
                1,
                "bitblas_tile_size":
                (self.bitblas_matmul.retrieve_weight_shape()[-2]
                 if self.bitblas_matmul.propagate_b else None),
                "pack_factor":
                self.quant_config.pack_factor,
                "weight_propagation":
                self.bitblas_matmul.propagate_b,
            },
        )

        sw = Parameter(
            torch.empty(
                1,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            sw,
            {
                "input_dim": None,
                "output_dim": None,
                "ignore_warning": True,
            },
        )
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("sw", sw)
        set_weight_attrs(sw, extra_weight_attrs)

    def create_weights_gptq(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Creates quantized weights for use in linear operations.

        The function initializes and returns a dictionary containing quantized 
        weights, scales, and zeros
        for performing quantized matrix multiplication operations.

        Args:
            input_size_per_partition: The size of the input partition.
            output_size_per_partition: The size of the output partition.
            input_size: The total size of the input (unused).
            output_size: The total size of the output (unused).
            params_dtype: 
                The data type of the parameters (expected to be torch.float16).

        Returns:
            A dictionary containing the quantized weights ('qweight'), 
            scales ('scales'), and zeros ('zeros').

        Raises:
            ValueError: If `params_dtype` is not `torch.float16` or if the 
            input size per partition is not divisible by the group size in 
            `quant_config`.
        """
        del input_size, output_size  # Unused arguments.
        if params_dtype != torch.float16:
            raise ValueError("Parameter data type must be torch.float16, "
                             f"but got {params_dtype}")
        group_size = self.quant_config.group_size
        if group_size is None:
            group_size = -1
        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if (group_size != -1 and input_size_per_partition % group_size != 0):
            raise ValueError(
                f"Input size per partition ({input_size_per_partition}) must "
                f"be divisible by group size ({group_size}).")

        # Initialize or retrieve the BitBLAS matrix multiplication operator.
        self._configure_bitblas_matmul(
            input_size_per_partition,
            output_size_per_partition,
            params_dtype=params_dtype,
            enable_tuning=self.ENABLE_TUNING,
            bias=False,
            layout="nt",
            bits=self.quant_config.weight_bits,
        )

        # Initialize quantized weights with dimensions

        qweight = Parameter(
            torch.empty(
                self.bitblas_matmul.retrieve_weight_shape(),
                device="cuda",
                dtype=self.quant_config.storage_torch_dtype,
            ),
            requires_grad=False,
        )
        # Attributes to help with unpacking and applying the weights later.
        set_weight_attrs(
            qweight,
            {
                "input_dim":
                1,
                "output_dim":
                0,
                "packed_dim":
                1,
                "bitblas_tile_size":
                (self.bitblas_matmul.retrieve_weight_shape()[-2]
                 if self.bitblas_matmul.propagate_b else None),
                "pack_factor":
                self.quant_config.pack_factor,
                "weight_propagation":
                self.bitblas_matmul.propagate_b,
            },
        )

        # Compute the number of input groups for channel-wise quantization.
        input_groups = (1 if group_size == -1 else input_size_per_partition //
                        group_size)

        # Initialize scales and zeros for the quantized weights.
        scales = Parameter(
            torch.empty(
                output_size_per_partition,
                input_groups,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": None if input_groups == 1 else 1,
            "output_dim": 0
        })
        if self.quant_config.zeros_mode == "quantized":
            zeros = Parameter(
                torch.empty(
                    input_groups,
                    output_size_per_partition // self.quant_config.pack_factor,
                    device="cuda",
                    dtype=self.quant_config.storage_torch_dtype,
                ),
                requires_grad=False,
            )
            # Set attributes to indicate how scales and zeros are applied.

            set_weight_attrs(
                zeros,
                {
                    "input_dim": None if input_groups == 1 else 0,
                    "output_dim": 1,
                    "packed_dim": 1,
                    "pack_factor": self.quant_config.pack_factor,
                },
            )
        else:
            zeros = Parameter(
                torch.empty(output_size_per_partition,
                            input_groups,
                            device="cuda",
                            dtype=params_dtype),
                requires_grad=False,
            )
            # Set attributes to indicate how scales and zeros are applied.
            set_weight_attrs(scales, {
                "input_dim": None if input_groups == 1 else 1,
                "output_dim": 0
            })

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("zeros", zeros)
        set_weight_attrs(zeros, extra_weight_attrs)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.quant_config.quant_method == "bitnet":
            return self.create_weights_bitnet(layer, input_size_per_partition,
                                              output_partition_sizes,
                                              input_size, output_size,
                                              params_dtype,
                                              **extra_weight_attrs)
        elif self.quant_config.quant_method == "gptq":
            return self.create_weights_gptq(layer, input_size_per_partition,
                                            output_partition_sizes, input_size,
                                            output_size, params_dtype,
                                            **extra_weight_attrs)
        else:
            raise ValueError(
                f"Unsupported quant_method {self.quant_config.quant_method}")

    def _configure_bitblas_matmul(
        self,
        infeatures,
        outfeatures,
        params_dtype,
        enable_tuning,
        bias,
        layout,
        bits,
        out_dtype="float16",
    ):
        from bitblas import MatmulConfig

        bitblas_dtype = self.BITBLAS_DTYPES[params_dtype]

        if self.quant_config.quant_method == "gptq":
            with_scaling = True
            with_zeros = True
            W_dtype = f"uint{bits}"
            group_size = self.quant_config.group_size
            zeros_mode = self.quant_config.zeros_mode
        elif self.quant_config.quant_method == "bitnet":
            with_scaling = False
            with_zeros = False
            W_dtype = f"int{bits}"
            group_size = None
            zeros_mode = None
        else:
            raise ValueError(
                f"Unsupported quant_method {self.quant_config.quant_method}")

        matmul_config = MatmulConfig(
            # M=self.OPT_FEATURES,
            N=outfeatures,
            K=infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=out_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.quant_config.STORAGE_DTYPE,
            with_scaling=with_scaling,
            with_zeros=with_zeros,
            group_size=group_size,
            with_bias=bias,
            layout=layout,
            zeros_mode=zeros_mode,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning)

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH,
                                                     BITBLAS_TARGET)

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                logger.info("BitBLAS Tuning done, appended operator to "
                            "global_operator_cache.")
            else:
                _message = f"BitBLAS Operator {config} created."
                logger.info(_message)
        else:
            _message = (
                f"BitBLAS Operator {config} found in global_operator_cache.")
            logger.info(_message)
        return bitblas_matmul

    @torch.compile
    def activation_quant(self, x):
        x = x.float()
        Qn = self.Qn
        Qp = self.Qp
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp).type(torch.int8)
        return result, s

    @torch.compile
    def post_quant_process(self, input, si, sw):
        out = input / si
        out = out / sw
        out = out.half()
        return out

    def apply_gptq(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.zeros

        x_2d = x.view(-1, x.shape[-1])

        output_2d = self.bitblas_matmul(x_2d, qweight, scales, qzeros)

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1], ))

        if bias is not None:
            output.add_(bias)  # In-place add

        return output

    def apply_bitnet(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        quant_input, si = self.activation_quant(x)

        output = self.bitblas_matmul(quant_input, layer.qweight)

        sw = layer.sw

        # if / (si * sw) will cause inf in some cases
        output = self.post_quant_process(output, si, sw)

        output = output.view(x.shape[:-1] + (output.shape[1], ))

        if bias is not None:
            output.add_(bias)  # In-place add

        return output

    def apply(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.quant_config.quant_method == "bitnet":
            return self.apply_bitnet(*args, **kwargs)
        elif self.quant_config.quant_method == "gptq":
            return self.apply_gptq(*args, **kwargs)
        else:
            raise ValueError(
                f"Unsupported quant_method {self.quant_config.quant_method}")
