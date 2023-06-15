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
      assert(input.device == noise.device)
      output = ptrnn_cuda.zmdrop_forward_cuda(input, noise, 1/(1-dropout_value))
    else:
      output = ptrnn_cpp.zmdrop_forward(input, noise, 1/(1-dropout_value))
    ctx.dropout_value = dropout_value
    ctx.save_for_backward(input)
    return output

  @staticmethod
  def backward(ctx, out_grad):
    (output,) = ctx.saved_variables
    if out_grad.is_cuda:
      assert(output.device == out_grad.device)
      in_grad = ptrnn_cuda.zmdrop_backward_cuda(output, out_grad, 1/(1-ctx.dropout_value))
    else:
      in_grad = ptrnn_cpp.zmdrop_backward(output, out_grad, 1/(1-ctx.dropout_value))
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
      return ZMDropoutFunction.apply(input, self.p)
    return input

def u2dt(t):
  if t.dim() == 2:
    return t.unsqueeze(0)
  else:
    return t

def dim23(f):
  def r(*args, **kwargs):
    args3d = [u2dt(t) for t in args]
    kwargs3d = {k:u2dt(v) for (k,v) in kwargs.items()}
    return f(*args3d, *kwargs3d)
  return r

@dim23
def tanh_gradient_mul(igrad, out, ograd1, ograd2):
  assert(igrad.dim() == 3)
  if igrad.is_cuda:
    ptrnn_cuda.tanh_gradient_mul_cuda(igrad, out, ograd1, ograd2)
  else:
    return ptrnn_cpp.tanh_gradient_mul(igrad, out, ograd1, ograd2)

@dim23
def tanh_gradient(igrad, out, ograd):
  if igrad.is_cuda:
    assert(igrad.dim() == 3)
    ptrnn_cuda.tanh_gradient_cuda(igrad, out, ograd)
  else:
    return ptrnn_cpp.tanh_gradient(igrad, out, ograd)

@dim23
def sigmoid_gradient_mul(igrad, out, ograd1, ograd2):
  if igrad.is_cuda:
    ptrnn_cuda.sigmoid_gradient_mul_cuda(igrad, out, ograd1, ograd2)
  else:
    return ptrnn_cpp.sigmoid_gradient_mul(igrad, out, ograd1, ograd2)

@dim23
def sigmoid_gradient(igrad, out, ograd):
  if igrad.is_cuda:
    ptrnn_cuda.sigmoid_gradient_cuda(igrad, out, ograd)
  else:
    return ptrnn_cpp.sigmoid_gradient(igrad, out, ograd)

@dim23
def u_gate(next_ht, prev_ht, u, hc):
  if u.is_cuda:
    return ptrnn_cuda.u_gate_cuda(next_ht, prev_ht, u, hc)
  else:
    return ptrnn_cpp.u_gate(next_ht, prev_ht, u, hc)
