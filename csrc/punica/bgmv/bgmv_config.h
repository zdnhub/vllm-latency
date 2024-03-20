#pragma once

template <int feat_in, int feat_out, typename in_T, typename out_T,
          typename W_T>
void bgmv_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                 const W_T *__restrict__ W,
                 const int64_t *__restrict__ indicies, int64_t y_offset,
                 int64_t full_y_size, int64_t batch_size, int64_t num_layers,
                 int64_t layer_idx, float scale);

// clang-format off

#define FOR_BGMV_WIDE(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, narrow, 128) \
    f(in_T, out_T, W_T, narrow, 256) \
    f(in_T, out_T, W_T, narrow, 512) \
    f(in_T, out_T, W_T, narrow, 768) \
    f(in_T, out_T, W_T, narrow, 1024) \
    f(in_T, out_T, W_T, narrow, 1280) \
    f(in_T, out_T, W_T, narrow, 1728) \
    f(in_T, out_T, W_T, narrow, 1792) \
    f(in_T, out_T, W_T, narrow, 2048) \
    f(in_T, out_T, W_T, narrow, 2560) \
    f(in_T, out_T, W_T, narrow, 2752) \
    f(in_T, out_T, W_T, narrow, 2816) \
    f(in_T, out_T, W_T, narrow, 3072) \
    f(in_T, out_T, W_T, narrow, 3456) \
    f(in_T, out_T, W_T, narrow, 3584) \
    f(in_T, out_T, W_T, narrow, 4096) \
    f(in_T, out_T, W_T, narrow, 5120) \
    f(in_T, out_T, W_T, narrow, 5504) \
    f(in_T, out_T, W_T, narrow, 5632) \
    f(in_T, out_T, W_T, narrow, 6144) \
    f(in_T, out_T, W_T, narrow, 6912) \
    f(in_T, out_T, W_T, narrow, 7168) \
    f(in_T, out_T, W_T, narrow, 8192) \
    f(in_T, out_T, W_T, narrow, 9216) \
    f(in_T, out_T, W_T, narrow, 10240) \
    f(in_T, out_T, W_T, narrow, 11008) \
    f(in_T, out_T, W_T, narrow, 12288) \
    f(in_T, out_T, W_T, narrow, 13696) \
    f(in_T, out_T, W_T, narrow, 13824) \
    f(in_T, out_T, W_T, narrow, 14336) \
    f(in_T, out_T, W_T, narrow, 16384) \
    f(in_T, out_T, W_T, narrow, 20480) \
    f(in_T, out_T, W_T, narrow, 22016) \
    f(in_T, out_T, W_T, narrow, 24576) \
    f(in_T, out_T, W_T, narrow, 28672) \
    f(in_T, out_T, W_T, narrow, 32000) \
    f(in_T, out_T, W_T, narrow, 32256) \
    f(in_T, out_T, W_T, narrow, 32512) \
    f(in_T, out_T, W_T, narrow, 32768) \
    f(in_T, out_T, W_T, narrow, 33024) \
    f(in_T, out_T, W_T, narrow, 36864) \
    f(in_T, out_T, W_T, narrow, 49152) \
// Keep above in sync with vllm/lora/layers::SamplerWithLoRA

// Used for defining kernels going from the variety of dim in to the narrow dim out
    // one, because the kernel has this flexibility so add support for it
    // but mainly, using it for the column parallel LoRA A from S-LoRA which splits the rank dim, but all_gathers before LoRA B 
#define FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, 128, narrow) \
    f(in_T, out_T, W_T, 256, narrow) \
    f(in_T, out_T, W_T, 512, narrow) \
    f(in_T, out_T, W_T, 768, narrow) \
    f(in_T, out_T, W_T, 1024, narrow) \
    f(in_T, out_T, W_T, 1280, narrow) \
    f(in_T, out_T, W_T, 1728, narrow) \
    f(in_T, out_T, W_T, 1792, narrow) \
    f(in_T, out_T, W_T, 2048, narrow) \
    f(in_T, out_T, W_T, 2560, narrow) \
    f(in_T, out_T, W_T, 2752, narrow) \
    f(in_T, out_T, W_T, 2816, narrow) \
    f(in_T, out_T, W_T, 3072, narrow) \
    f(in_T, out_T, W_T, 3456, narrow) \
    f(in_T, out_T, W_T, 3584, narrow) \
    f(in_T, out_T, W_T, 4096, narrow) \
    f(in_T, out_T, W_T, 5120, narrow) \
    f(in_T, out_T, W_T, 5504, narrow) \
    f(in_T, out_T, W_T, 5632, narrow) \
    f(in_T, out_T, W_T, 6144, narrow) \
    f(in_T, out_T, W_T, 6912, narrow) \
    f(in_T, out_T, W_T, 7168, narrow) \
    f(in_T, out_T, W_T, 8192, narrow) \
    f(in_T, out_T, W_T, 9216, narrow) \
    f(in_T, out_T, W_T, 10240, narrow) \
    f(in_T, out_T, W_T, 11008, narrow) \
    f(in_T, out_T, W_T, 12288, narrow) \
    f(in_T, out_T, W_T, 13824, narrow) \
    f(in_T, out_T, W_T, 13696, narrow) \
    f(in_T, out_T, W_T, 14336, narrow) \
    f(in_T, out_T, W_T, 16384, narrow) \
    f(in_T, out_T, W_T, 20480, narrow) \
    f(in_T, out_T, W_T, 22016, narrow) \
    f(in_T, out_T, W_T, 24576, narrow) \
    f(in_T, out_T, W_T, 28672, narrow) \
    f(in_T, out_T, W_T, 32000, narrow) \
    f(in_T, out_T, W_T, 32256, narrow) \
    f(in_T, out_T, W_T, 32512, narrow) \
    f(in_T, out_T, W_T, 32768, narrow) \
    f(in_T, out_T, W_T, 33024, narrow) \
    f(in_T, out_T, W_T, 36864, narrow) \
    f(in_T, out_T, W_T, 49152, narrow) \
// Keep above in sync with vllm/lora/layers::SamplerWithLoRA


// Keep this in sync with vllm/config::LoRAConfig
#define FOR_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 8)  \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 16) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 32) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 64)


#define FOR_INST_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 1) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 2) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 4) \

// clang-format on
