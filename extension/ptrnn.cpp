#include <torch/extension.h>

#include <iostream>

torch::Tensor zmdrop_forward(torch::Tensor input, torch::Tensor noise, float mult) {
  auto input_a = input.accessor<float, 3>();
  auto noise_a = noise.accessor<uint8_t, 3>();
  TORCH_CHECK(!input.is_cuda());
  TORCH_CHECK(!noise.is_cuda());
  int N = input_a.size(0);
  int T = input_a.size(1);
  int D = input_a.size(2);
  TORCH_CHECK(noise_a.size(0) == N);
  TORCH_CHECK(noise_a.size(1) == T);
  TORCH_CHECK(noise_a.size(2) == D);
  for (int n = 0; n < N; n++) {
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
  return input;
}

torch::Tensor zmdrop_backward(torch::Tensor input, torch::Tensor grad, float mult) {
  auto input_a = input.accessor<float, 3>();
  auto grad_a = grad.accessor<float, 3>();
  TORCH_CHECK(!input.is_cuda());
  TORCH_CHECK(!grad.is_cuda());
  int N = input_a.size(0);
  int T = input_a.size(1);
  int D = input_a.size(2);
  TORCH_CHECK(grad_a.size(0) == N);
  TORCH_CHECK(grad_a.size(1) == T);
  TORCH_CHECK(grad_a.size(2) == D);
  for (int n = 0; n < N; n++) {
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
  return grad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zmdrop_forward", &zmdrop_forward, "ZMDropout forward");
  m.def("zmdrop_backward", &zmdrop_backward, "ZMDropout backward");
}
