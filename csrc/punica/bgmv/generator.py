DTYPES = ["fp16", "bf16", "fp32"]
DTYPE_MAP = {
    "fp16": "nv_half",
    "bf16": "nv_bfloat16",
    "fp32": "float",
}

TEMPLATE = """
#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, {input_dtype}, {output_dtype}, {weight_dtype})
FOR_INST_BGMV_WIDE_NARROW(INST_BGMV_ONESIDE, {input_dtype}, {output_dtype}, {weight_dtype})
""".lstrip()

for input_dtype in DTYPES:
    for output_dtype in DTYPES:
        for weight_dtype in DTYPES:
            if weight_dtype == "fp32":
                # FP32 weights are not supported.
                continue
            if output_dtype == "fp32":
                # LoRA A matrix.
                if input_dtype != weight_dtype:
                    # NOTE(woosuk): While Punica supports the case where the
                    # input and weight dtypes are different, we only generate
                    # the kernels the same dtypes to reduce the binary size.
                    continue
            elif input_dtype == "fp32":
                # LoRA B matrix.
                if output_dtype != weight_dtype:
                    # NOTE(woosuk): While Punica supports the case where the
                    # output and weight dtypes are different, we only generate
                    # the kernels the same dtypes to reduce the binary size.
                    continue
            elif not (input_dtype == output_dtype == weight_dtype):
                # NOTE(woosuk): While Punica supports mixed data types for
                # input, output, and weight, we only generate the kernels with
                # the same data types to reduce the binary size.
                continue

            kernel_definition = TEMPLATE.format(
                input_dtype=DTYPE_MAP[input_dtype],
                output_dtype=DTYPE_MAP[output_dtype],
                weight_dtype=DTYPE_MAP[weight_dtype])
            filename = f"bgmv_{input_dtype}_{output_dtype}_{weight_dtype}.cu"
            with open(filename, "w") as f:
                f.write(kernel_definition)
