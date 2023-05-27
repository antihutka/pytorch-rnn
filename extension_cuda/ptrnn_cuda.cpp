#include <torch/extension.h>
#include <iostream>

torch::Tensor zmdrop_forward_cuda(torch::Tensor input, torch::Tensor noise, float mult);
torch::Tensor zmdrop_backward_cuda(torch::Tensor input, torch::Tensor grad, float mult);
torch::Tensor tanh_gradient_cuda(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd);
torch::Tensor tanh_gradient_mul_cuda(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd1, torch::Tensor ograd2);
torch::Tensor sigmoid_gradient_cuda(torch::Tensor igrad, torch::Tensor out, torch::Tensor ograd);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("zmdrop_forward_cuda", &zmdrop_forward_cuda, "ZMDropout forward CUDA");
  m.def("zmdrop_backward_cuda", &zmdrop_backward_cuda, "ZMDropout backward CUDA");
  m.def("tanh_gradient_cuda", &tanh_gradient_cuda, "tanh gradient CUDA");
  m.def("tanh_gradient_mul_cuda", &tanh_gradient_mul_cuda, "tanh gradient CUDA with ograd mul");
  m.def("sigmoid_gradient_cuda", &sigmoid_gradient_cuda, "sigmoid gradient CUDA");
}
