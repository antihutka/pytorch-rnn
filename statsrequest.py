import time
import sampling

class StatsRequestModule():
  def __init__(self, sampler):
    self.sampler = sampler
  def forward(self, request):
    request.start_time = time.clock()
  def backward(self, request):
    request.end_time = time.clock()
    request.elapsed = request.end_time - request.start_time

class StatsRequest(sampling.SamplerRequest):
  def __init__(self, sampler):
    self.chains = sampling.SamplerChains([StatsRequestModule(sampler)], [], [])
    self.key = None
    self.samples = []

