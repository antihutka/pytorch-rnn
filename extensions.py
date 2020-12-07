import torch
import ptrnn_cpp
if torch.cuda.is_available():
  import ptrnn_cuda

class ZMDropoutFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, dropout_value):
    noise = input.new_empty(input.size(), dtype=torch.uint8)
    noise.bernoulli_(1-dropout_value)
    if input.is_cuda:
      output = ptrnn_cuda.zmdrop_forward_cuda(input, noise)
    else:
      output = ptrnn_cpp.zmdrop_forward(input, noise)
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, out_grad):
    (output,) = ctx.saved_variables
    if out_grad.is_cuda:
      in_grad = ptrnn_cuda.zmdrop_backward_cuda(output, out_grad)
    else:
      in_grad = ptrnn_cpp.zmdrop_backward(output, out_grad)
    return in_grad, None

class ZMDropout(torch.nn.Module):
  def __init__(self, p, inplace):
    super(ZMDropout, self).__init__()
    assert(inplace)
    if p < 0 or p > 1:
      raise ValueError("dropout probability has to be between 0 and 1, "
                       "but got {}".format(p))
    if not inplace:
      raise ValueError("ZMDropout requires inplace=True")
    self.p = p
  def forward(self, input):
    if self.training:
      ZMDropoutFunction.apply(input, self.p)
      input.mul_(1/(1-self.p))
    return input
