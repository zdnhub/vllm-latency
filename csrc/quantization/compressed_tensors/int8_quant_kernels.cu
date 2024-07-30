#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#include "../../dispatch_utils.h"
#include "../../reduction_utils.cuh"

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static const float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static const float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  // round
  float dst = std::nearbyint(x);
  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const* scale_ptr, const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  scale_type const scale = *scale_ptr;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] = float_to_int8_rn(
        static_cast<float>(input[token_idx * hidden_size + i]) / scale);
  }
}

template <typename scalar_t, typename scale_type>
__global__ void dynamic_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(input[token_idx * hidden_size + i]);
    val = val > zero ? val : -val;
    absmax_val = val > absmax_val ? val : absmax_val;
  }

  float const block_absmax_val_maybe = blockReduceMax(absmax_val);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / 127.0f;
  }
  __syncthreads();

  float const tmp_scale = 127.0f / block_absmax_val;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] = float_to_int8_rn(
        static_cast<float>(input[token_idx * hidden_size + i]) * tmp_scale);
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, azp_type* azp, const int hidden_size) {
  int const token_idx = blockIdx.x;

  // Scan for the min and max value for this token
  float max_val = 0.0f;
  float min_val = std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto val = static_cast<float>(input[token_idx * hidden_size + i]);
    max_val = fmaxf(max_val, val);
    min_val = fminf(min_val, val);
  }

  // Reduce the max and min values across the block
  max_val = blockReduceMax(max_val);
  //  if (threadIdx.x == 0 and blockIdx.x == DEBUG_TOKEN) printf("MIN:");
  min_val = blockReduceMin(min_val);

  __shared__ scale_type scale_sh;
  __shared__ azp_type azp_sh;

  // Compute the scale and zero point and store them, only on the first thread
  if (threadIdx.x == 0) {
    float scale_val = (max_val - min_val) / 255.0f;
    auto const azp_float = roundf(min_val / scale_val + 128.0f);
    auto const azp_val = static_cast<azp_type>(azp_float);

    // Azp was rounded, which may cause the range to be slightly off.
    // Expand the range to make sure all values are representable.
    auto const min_nozp = static_cast<float>(azp_val - 128);
    auto const max_nozp = static_cast<float>(azp_val + 127);
    auto no_div_0 = [&](float num, float div) {
      return div == 0.0f ? scale_val : num / div;
    };

    scale_val = fmaxf(no_div_0(max_val, max_nozp), no_div_0(min_val, min_nozp));

    // Store the scale and azp
    scale[token_idx] = scale_sh = scale_val;
    azp[token_idx] = azp_sh = azp_val;
  }

  // Wait for the scale and azp to be computed
  __syncthreads();

  float const scale_val = scale_sh;
  azp_type const azp_val = azp_sh;

  // Quantize the values
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto val = static_cast<float>(input[token_idx * hidden_size + i]);
    auto quant_val = static_cast<int8_t>(roundf(val / scale_val) - azp_val);
    out[token_idx * hidden_size + i] = quant_val;
  }
}

}  // namespace vllm

void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              torch::Tensor const& input,  // [..., hidden_size]
                              torch::Tensor const& scale) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scale.numel() == 1);

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_kernel", [&] {
        vllm::static_scaled_int8_quant_kernel<scalar_t, float>
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),
                                         out.data_ptr<int8_t>(),
                                         scale.data_ptr<float>(), hidden_size);
      });
}

void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales, c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          vllm::dynamic_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scales.data_ptr<float>(), hidden_size);
        } else {
          vllm::dynamic_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scales.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size);
        }
      });
}
