#include <torch/extension.h>
#include <iostream>

torch::Tensor zmdrop_forward_cuda(torch::Tensor input, torch::Tensor noise, float mult);
torch::Tensor zmdrop_backward_cuda(torch::Tensor input, torch::Tensor grad, float mult);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zmdrop_forward_cuda", &zmdrop_forward_cuda, "ZMDropout forward CUDA");
  m.def("zmdrop_backward_cuda", &zmdrop_backward_cuda, "ZMDropout backward CUDA");
}
