import torch
import ptrnn_cpp

class BetterAdamW(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay = 0.0, grad_clip = 0.0):
    defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay, 'grad_clip':grad_clip}
    super().__init__(params, defaults)

  def create_state_for(self, p):
    state = self.state[p]
    if len(state) == 0:
      state["step"] = 0
      state["exp_avg"] = torch.zeros_like(p, device='cpu')
      state["exp_avg_sq"] = torch.zeros_like(p, device='cpu')
      if p.is_cuda:
        state["param_cache"] = torch.empty_like(p, device='cpu', pin_memory=True).copy_(p.data, non_blocking=True)
        state["grad_cache"] = torch.empty_like(p, device='cpu', pin_memory=True)
    return state

  def register_hooks(self):
    stream = torch.cuda.Stream()
    @torch.no_grad()
    def hook(p):
      state = self.create_state_for(p)
      event1 = torch.cuda.Event();
      event1.record()
      with torch.cuda.stream(stream):
        event1.wait()
        state["grad_cache"].copy_(p.grad, non_blocking=True)
        state["event"] = torch.cuda.Event()
        state["event"].record()
        p.grad.record_stream(stream)
      p.grad = None
    for group in self.param_groups:
      for p in group["params"]:
        if p.is_cuda and p.requires_grad:
          p.register_post_accumulate_grad_hook(hook)

  @torch.no_grad()
  def step(self, closure=None):
    if closure is not None:
      with torch.enable_grad():
        closure()
    for group in self.param_groups:
      for p in group["params"]:
        state = self.create_state_for(p)
        if p.is_cuda and "event" not in state and p.grad is not None:
          state["grad_cache"].copy_(p.grad, non_blocking=True)
          state["event"] = torch.cuda.Event()
          state["event"].record()
    for group in self.param_groups:
      beta1, beta2 = group["betas"]
      lr = group["lr"]
      weight_decay = group["weight_decay"]
      eps = group["eps"]
      grad_clip = group["grad_clip"]
      for p in group["params"]:
        state = self.state[p]
        if p.grad is None and "event" not in state:
          continue
        if p.grad is not None and p.grad.is_sparse:
          raise RuntimeError("BetterAdamW does not support sparse gradients")
        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if p.is_cuda:
          param = state["param_cache"]
          grad = state["grad_cache"]
          state["event"].synchronize()
          state["event"]=None
        else:
          param = p.data
          grad = p.grad
        ptrnn_cpp.adamW_step(param, grad, exp_avg, exp_avg_sq, lr, weight_decay, eps, beta1, beta2, grad_clip, step)
        if p.is_cuda:
          p.data.copy_(param, non_blocking=True)
