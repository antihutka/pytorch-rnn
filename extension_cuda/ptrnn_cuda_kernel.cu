#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define TPB 256

template <typename scalar_t> __global__ void zmdrop_forward_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
  torch::PackedTensorAccessor32<uint8_t,3,torch::RestrictPtrTraits> noise,
  int D, float mult) {
  int n = blockIdx.x;
  int t = blockIdx.y;
  int d = TPB * blockIdx.z + threadIdx.x;
  if (d < D) {
    scalar_t nv = input[n][t][d];
    nv *= mult;
    if (noise[n][t][d] == 0)
      nv = 0;
    else if (nv == 0)
      nv = 1e-45;
    input[n][t][d] = nv;
  }
}

torch::Tensor zmdrop_forward_cuda(torch::Tensor input, torch::Tensor noise, float mult) {
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(noise.is_cuda());
  int N = input.size(0);
  int T = input.size(1);
  int D = input.size(2);
  TORCH_CHECK(noise.size(0) == N);
  TORCH_CHECK(noise.size(1) == T);
  TORCH_CHECK(noise.size(2) == D);
  const dim3 threads(TPB);
  const dim3 blocks(N, T, (D+TPB-1)/TPB);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "zmdrop_forward_cuda", ([&] { 
    zmdrop_forward_kernel<scalar_t><<<blocks, threads>>>(
      input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      noise.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
      D, mult);
  }));
  return input;
}

template <typename scalar_t> __global__ void zmdrop_backward_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad,
  int D, float mult) {
  int n = blockIdx.x;
  int t = blockIdx.y;
  int d = TPB * blockIdx.z + threadIdx.x;
  if (d < D) {
    scalar_t g = grad[n][t][d];
//    g *= mult;
    if (input[n][t][d] == 0)
      g = 0;
    else
      g *= mult;
    grad[n][t][d] = g;
  }
}

torch::Tensor zmdrop_backward_cuda(torch::Tensor input, torch::Tensor grad, float mult) {
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(grad.is_cuda());
  int N = input.size(0);
  int T = input.size(1);
  int D = input.size(2);
  TORCH_CHECK(grad.size(0) == N);
  TORCH_CHECK(grad.size(1) == T);
  TORCH_CHECK(grad.size(2) == D);
  const dim3 threads(TPB);
  const dim3 blocks(N, T, (D+TPB-1)/TPB);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "zmdrop_backward_cuda", ([&] { 
    zmdrop_backward_kernel<scalar_t><<<blocks, threads>>>(
      input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      D, mult);
  }));
  return grad;
}
