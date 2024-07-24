#pragma once

/**
 * __device__ layernorm utilities.
 */

#include "vectorization.cuh"
#include "quant_conversions.cuh"

namespace vllm {

// has_residual must be true, if residual is not a nullptr
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;
  // sum of squares
  float ss = 0.0f;

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = (float)input[token_offset + i];
    if constexpr (has_residual) {
      x += (float)residual[token_offset + i];
    }

    ss += x * x;
  }
  ss = blockReduceSum<float>(ss);
  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    float const min_scaling_factor, int const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;
  constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

  float block_absmax_val_maybe = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = (float)input[token_offset + i];
    if constexpr (has_residual) {
      x += (float)residual[token_offset + i];
    }

    x = x * rms * (float)(weight[i]);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
  }
  block_absmax_val_maybe = blockReduceMax(block_absmax_val_maybe);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor);
    s_token_scale = scale;                 // Shared memory store
    all_token_scales[blockIdx.x] = scale;  // Global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float const w = (float)weight[i];
    float x = (float)input[token_offset + i];
    if constexpr (has_residual) {
      x += (float)residual[token_offset + i];
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = x * rms * w;
    // Quant
    output[i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale);
  }
}

namespace vectorized {

// Compute 1.0/rms(input)
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;

  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  // sum of squares
  float ss = 0.0f;

  int const tid = threadIdx.x;
  int const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.x);
      x.z += static_cast<float>(r.x);
      x.w += static_cast<float>(r.x);
    }

    ss += x.x * x.x;
    ss += x.y * x.y;
    ss += x.z * x.z;
    ss += x.w * x.w;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int i = num_vec_elems * 4 + tid; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }
    ss += x * x;
  }

  ss = blockReduceSum<float>(ss);
  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

// Vectorized version of vllm::compute_dynamic_per_token_scales
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    float const min_scaling_factor, int const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;

  // Vectorized input/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};
  int const tid = threadIdx.x;

  int const num_vec_elems = hidden_size >> 2;
  float block_absmax_val_maybe = 0.0f;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
    }

    block_absmax_val_maybe =
        fmaxf(block_absmax_val_maybe, x.x * rms * (float)(w.x));
    block_absmax_val_maybe =
        fmaxf(block_absmax_val_maybe, x.y * rms * (float)(w.y));
    block_absmax_val_maybe =
        fmaxf(block_absmax_val_maybe, x.z * rms * (float)(w.z));
    block_absmax_val_maybe =
        fmaxf(block_absmax_val_maybe, x.w * rms * (float)(w.w));
  }

  for (int i = num_vec_elems * 4 + tid; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }
    block_absmax_val_maybe =
        fmaxf(block_absmax_val_maybe, x * rms * (float)(weight[i]));
  }

  block_absmax_val_maybe = blockReduceMax(block_absmax_val_maybe);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor);
    s_token_scale = scale;                 // shared memory store
    all_token_scales[blockIdx.x] = scale;  // global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  int const tid = threadIdx.x;
  int const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> const in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
      // Update residual
      r.x = static_cast<scalar_t>(x.x);
      r.y = static_cast<scalar_t>(x.y);
      r.z = static_cast<scalar_t>(x.z);
      r.w = static_cast<scalar_t>(x.w);
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;
    out.x = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        x.x * rms * w.x, scale);
    out.y = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        x.y * rms * w.y, scale);
    out.z = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        x.z * rms * w.z, scale);
    out.w = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        x.w * rms * w.w, scale);
    vec_output[i] = out;
  }

  for (int i = num_vec_elems * 4 + tid; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    output[i] = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        x * rms * weight[i], scale);
  }
}

}  // namespace vectorized

}  // namespace vllm
