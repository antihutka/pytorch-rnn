#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/DeviceGuard.h>

#define TPB 256
#define GETNTDSIZE(x) int N=x.size(0), T=x.size(1), D=x.size(2)
#define ISCUDA(x) TORCH_CHECK(x.is_cuda())
#define ISCUDA2(x,y) TORCH_CHECK(x.is_cuda() && y.is_cuda())
#define ISCUDA3(x,y,z) TORCH_CHECK(x.is_cuda() && y.is_cuda() && z.is_cuda())
#define ISCUDA4(x, y, z, w) TORCH_CHECK(x.is_cuda() && y.is_cuda() && z.is_cuda() && w.is_cuda())
#define SAMESIZE(x,y) TORCH_CHECK(x.size(0)==y.size(0) && x.size(1)==y.size(1) && x.size(2) == y.size(2))
#define CALCBT const dim3 threads(TPB), blocks(N, T, (D+TPB-1)/TPB)
#define CALCNTD int n=blockIdx.x, t=blockIdx.y, d=TPB*blockIdx.z+threadIdx.x

template <typename scalar_t> __global__ void zmdrop_forward_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
  torch::PackedTensorAccessor32<uint8_t,3,torch::RestrictPtrTraits> noise,
  int D, float mult) {
  CALCNTD;
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
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  ISCUDA2(input, noise);
  GETNTDSIZE(input);
  SAMESIZE(input, noise);
  CALCBT;
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
  CALCNTD;
  if (d < D) {
    scalar_t g = grad[n][t][d];
    if (input[n][t][d] == 0)
      g = 0;
    else
      g *= mult;
    grad[n][t][d] = g;
  }
}

torch::Tensor zmdrop_backward_cuda(torch::Tensor input, torch::Tensor grad, float mult) {
  const c10::OptionalDeviceGuard device_guard(device_of(input));
  ISCUDA2(input, grad)
  GETNTDSIZE(input);
  SAMESIZE(input, grad);
  CALCBT;
  AT_DISPATCH_FLOATING_TYPES(input.type(), "zmdrop_backward_cuda", ([&] { 
    zmdrop_backward_kernel<scalar_t><<<blocks, threads>>>(
      input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      D, mult);
  }));
  return grad;
}

template <typename scalar_t> __global__ void tanh_gradient_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> igrad,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ograd,
  int D) {
  CALCNTD;
  if (d < D) {
    scalar_t o = out[n][t][d];
    igrad[n][t][d] = ograd[n][t][d] * (1 - o*o);
  }
}

torch::Tensor tanh_gradient_cuda(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd) {
  ISCUDA3(igrad, out, ograd)
  GETNTDSIZE(igrad);
  SAMESIZE(igrad, out);
  SAMESIZE(igrad, ograd);
  CALCBT;
  AT_DISPATCH_FLOATING_TYPES(igrad.type(), "tanh_gradient_cuda", ([&] {
    tanh_gradient_kernel<scalar_t><<<blocks, threads>>>(
      igrad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      ograd.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      D);
  }));
  return igrad;
}

template <typename scalar_t> __global__ void tanh_gradient_mul_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> igrad,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ograd1,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ograd2,
  int D) {
  CALCNTD;
  if (d < D) {
    scalar_t o = out[n][t][d];
    igrad[n][t][d] = ograd1[n][t][d] * ograd2[n][t][d] * (1 - o*o);
  }
}

torch::Tensor tanh_gradient_mul_cuda(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd1, torch::Tensor ograd2) {
  ISCUDA4(igrad, out, ograd1, ograd2)
  GETNTDSIZE(igrad);
  SAMESIZE(igrad, out);
  SAMESIZE(igrad, ograd1);
  SAMESIZE(igrad, ograd2);
  CALCBT;
  AT_DISPATCH_FLOATING_TYPES(igrad.type(), "tanh_gradient_cuda", ([&] {
    tanh_gradient_mul_kernel<scalar_t><<<blocks, threads>>>(
      igrad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      ograd1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      ograd2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      D);
  }));
  return igrad;
}


template <typename scalar_t> __global__ void sigmoid_gradient_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> igrad,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ograd,
  int D) {
  CALCNTD;
  if (d < D) {
    scalar_t o = out[n][t][d];
    igrad[n][t][d] = ograd[n][t][d] * o * (1-o);
  }
}

torch::Tensor sigmoid_gradient_cuda(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd) {
  ISCUDA3(igrad, out, ograd)
  GETNTDSIZE(igrad);
  SAMESIZE(igrad, out);
  SAMESIZE(igrad, ograd);
  CALCBT;
  AT_DISPATCH_FLOATING_TYPES(igrad.type(), "sigmoid_gradient_cuda", ([&] {
    sigmoid_gradient_kernel<scalar_t><<<blocks, threads>>>(
      igrad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      ograd.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      D);
  }));
  return igrad;
}
