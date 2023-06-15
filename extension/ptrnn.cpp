#include <torch/extension.h>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>

#include <iostream>

#define SAMESIZE(x,y) TORCH_CHECK(x.size(0)==y.size(0) && x.size(1)==y.size(1) && x.size(2) == y.size(2))

torch::Tensor zmdrop_forward(torch::Tensor input, torch::Tensor noise, float mult) {
  TORCH_CHECK(!input.is_cuda());
  TORCH_CHECK(!noise.is_cuda());
  SAMESIZE(input, noise);
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.type(), "zmdrop_forward", ([&] {
    auto input_a = input.accessor<scalar_t, 3>();
    auto noise_a = noise.accessor<uint8_t, 3>();
    int N = input_a.size(0);
    int T = input_a.size(1);
    int D = input_a.size(2);
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int n = start; n < end; n++) {
        for (int t = 0; t < T; t++) {
          for (int d = 0; d < D; d++) {
            if (noise_a[n][t][d] == 0) {
              input_a[n][t][d] = 0;
            } else if (input_a[n][t][d] == 0) {
              input_a[n][t][d] = 1e-45;
            } else {
              input_a[n][t][d] *= mult;
            }
          }
        }
      }
    });
  }));
  return input;
}

torch::Tensor zmdrop_backward(torch::Tensor input, torch::Tensor grad, float mult) {
  TORCH_CHECK(!input.is_cuda());
  TORCH_CHECK(!grad.is_cuda());
  SAMESIZE(input, grad);
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.type(), "zmdrop_backward", ([&] {
    auto input_a = input.accessor<scalar_t, 3>();
    auto grad_a = grad.accessor<scalar_t, 3>();
    int N = input_a.size(0);
    int T = input_a.size(1);
    int D = input_a.size(2);
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int n = start; n < end; n++) {
        for (int t = 0; t < T; t++) {
          for (int d = 0; d < D; d++) {
            if (input_a[n][t][d] == 0) {
              grad_a[n][t][d] = 0;
            } else {
              grad_a[n][t][d] *= mult;
            }
          }
        }
      }
    });
  }));
  return grad;
}

torch::Tensor sigmoid_gradient(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd) {
  TORCH_CHECK(!igrad.is_cuda() && !out.is_cuda() && !ograd.is_cuda())
  SAMESIZE(igrad, out)
  SAMESIZE(ograd, out)
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, igrad.type(), "sigmoid_gradient", ([&] {
    auto igrad_a = igrad.accessor<scalar_t, 3>();
    auto out_a = out.accessor<scalar_t, 3>();
    auto ograd_a = ograd.accessor<scalar_t, 3>();
    int N = igrad_a.size(0);
    int T = out_a.size(1);
    int D = ograd_a.size(2);
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int n = start; n < end; n++) {
        for (int t = 0; t < T; t++) {
          for (int d = 0; d < D; d++) {
            scalar_t o = out_a[n][t][d];
            igrad_a[n][t][d] = ograd_a[n][t][d] * o * (1-o);
          }
        }
      }
    });
  }));
  return igrad;
}

torch::Tensor sigmoid_gradient_mul(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd1, torch::Tensor ograd2) {
  TORCH_CHECK(!igrad.is_cuda() && !out.is_cuda() && !ograd1.is_cuda() && !ograd2.is_cuda())
  SAMESIZE(igrad, out)
  SAMESIZE(ograd1, out)
  SAMESIZE(ograd2, out)
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, igrad.type(), "sigmoid_gradient_mul", ([&] {
    auto igrad_a = igrad.accessor<scalar_t, 3>();
    auto out_a = out.accessor<scalar_t, 3>();
    auto ograd1_a = ograd1.accessor<scalar_t, 3>();
    auto ograd2_a = ograd2.accessor<scalar_t, 3>();
    int N = igrad_a.size(0);
    int T = out_a.size(1);
    int D = ograd1_a.size(2);
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int n = start; n < end; n++) {
        for (int t = 0; t < T; t++) {
          for (int d = 0; d < D; d++) {
            scalar_t o = out_a[n][t][d];
            igrad_a[n][t][d] = ograd1_a[n][t][d] * ograd2_a[n][t][d] * o * (1-o);
          }
        }
      }
    });
  }));
  return igrad;
}

torch::Tensor tanh_gradient(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd) {
  TORCH_CHECK(!igrad.is_cuda() && !out.is_cuda() && !ograd.is_cuda())
  SAMESIZE(igrad, out)
  SAMESIZE(ograd, out)
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, igrad.type(), "tanh_gradient", ([&] {
    auto igrad_a = igrad.accessor<scalar_t, 3>();
    auto out_a = out.accessor<scalar_t, 3>();
    auto ograd_a = ograd.accessor<scalar_t, 3>();
    int N = igrad_a.size(0);
    int T = out_a.size(1);
    int D = ograd_a.size(2);
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int n = start; n < end; n++) {
        for (int t = 0; t < T; t++) {
          for (int d = 0; d < D; d++) {
            scalar_t o = out_a[n][t][d];
            igrad_a[n][t][d] = ograd_a[n][t][d] * (1-o*o);
          }
        }
      }
    });
  }));
  return igrad;
}

torch::Tensor tanh_gradient_mul(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd1, torch::Tensor ograd2) {
  TORCH_CHECK(!igrad.is_cuda() && !out.is_cuda() && !ograd1.is_cuda() && !ograd2.is_cuda())
  SAMESIZE(igrad, out)
  SAMESIZE(ograd1, out)
  SAMESIZE(ograd2, out)
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, igrad.type(), "tanh_gradient_mul", ([&] {
    auto igrad_a = igrad.accessor<scalar_t, 3>();
    auto out_a = out.accessor<scalar_t, 3>();
    auto ograd1_a = ograd1.accessor<scalar_t, 3>();
    auto ograd2_a = ograd2.accessor<scalar_t, 3>();
    int N = igrad_a.size(0);
    int T = out_a.size(1);
    int D = ograd1_a.size(2);
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int n = start; n < end; n++) {
        for (int t = 0; t < T; t++) {
          for (int d = 0; d < D; d++) {
            scalar_t o = out_a[n][t][d];
            igrad_a[n][t][d] = ograd1_a[n][t][d] * ograd2_a[n][t][d] * (1-o*o);
          }
        }
      }
    });
  }));
  return igrad;
}

torch::Tensor u_gate(torch::Tensor next_ht, torch::Tensor prev_ht, torch::Tensor u, torch::Tensor hc) {
  TORCH_CHECK(!prev_ht.is_cuda() && !next_ht.is_cuda() && !u.is_cuda() && !hc.is_cuda())
  SAMESIZE(u, hc)
  SAMESIZE(u, next_ht)
  SAMESIZE(u, prev_ht)
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, u.type(), "u_gate", ([&] {
    auto next_ht_a = next_ht.accessor<scalar_t, 3>();
    auto prev_ht_a = prev_ht.accessor<scalar_t, 3>();
    auto u_a = u.accessor<scalar_t, 3>();
    auto hc_a = hc.accessor<scalar_t, 3>();
    int N = u_a.size(0);
    int T = u_a.size(1);
    int D = u_a.size(2);
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      for (int n = start; n < end; n++) {
        for (int t = 0; t < T; t++) {
          for (int d = 0; d < D; d++) {
            next_ht_a[n][t][d] = prev_ht_a[n][t][d] * (1-u_a[n][t][d]) + hc_a[n][t][d] * u_a[n][t][d];
          }
        }
      }
    });
  }));
  return next_ht;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zmdrop_forward", &zmdrop_forward, "ZMDropout forward");
  m.def("zmdrop_backward", &zmdrop_backward, "ZMDropout backward");
  m.def("sigmoid_gradient", &sigmoid_gradient, "Sigmoid gradient");
  m.def("sigmoid_gradient_mul", &sigmoid_gradient_mul, "Sigmoid gradient with output mul");
  m.def("tanh_gradient", &tanh_gradient, "Tanh gradient");
  m.def("tanh_gradient_mul", &tanh_gradient_mul, "Tanh gradient with output mul");
  m.def("u_gate", &u_gate, "U gate for gridgru");
}
