import modules as M
import torch
import util

def probs_from_scores(x):
  x = x[-1, :]
  x = x.double()
  x.exp_()
  x.div_(x.sum(0).unsqueeze(0))
  return x

def sample_from_probs(p):
  return torch.multinomial(p, 1)

# default chains:
# - request chain, get:
#   - DefaultStateStore (loads and saves states inside itself)
#   - SimpleSampling (converts Request into one Sample)
# - sampling chain, pre, get:
#   - PrepareInput (prepares input for model)
# - sampling chain, post, get:
#   - ApplyTemperature (scales Sample output by temperature)
#   - CalculateProbs (calculate probs from scores)
#   - SampleToken (samples token from probs)
#   - CheckEndingToken (check last sampled token, set finished flag)
#   - HardLengthLimit (check number of generated tokens, set finished flag)

# - request chain, put:
#   - DefaultStateStore
#   - SimpleSampling
# - sampling chain, pre, put:
#   - PrepareInput
# - post
#   - GetForcedInput (picks the next forced, sets finished flag as needed)

class SamplerRequest:
  def __init__(self, key, chains):
    self.chains = chains
    self.key = key
  def run_inchain(self):
    for mod in self.chains.request_chain:
      mod.forward(self)
  def run_outchain(self):
    assert all((s.finished for s in self.samples))
    for mod in reversed(self.chains.request_chain):
      mod.backward(self)

class Sample:
  def __init__(self, request, chains, initial_state, initial_token):
    self.request = request
    self.chains = chains
    self.sampled_sequence = []
    self.input_tokens = [initial_token]
    self.states = [initial_state]
    self.probs = []
    self.finished = False
  def token_add(self, token, probs, state):
    #print('adding token ', token)
    self.sampled_sequence += [token]
    self.input_tokens += [token]
    self.states += [state]
    self.probs += [probs]
  def token_del(self, cnt, soft_cnt = False):
    if soft_cnt:
      cnt = min(
        cnt,
        len(self.sampled_sequence),
        len(self.input_tokens) - 1,
        len(self.states) - 1,
        len(self.probs))
    assert (cnt > 0 and len(self.sampled_sequence) >= cnt)
    del self.sampled_sequence[-cnt:]
    del self.input_tokens[-cnt:]
    del self.states[-cnt:]
    del self.probs[-cnt:]

def fl(l):
  return "[" + ", ".join((str(i) for i in l)) + "]"

class SamplerChains():
  def __init__(self, request_chain, sample_pre, sample_post):
    self.request_chain = request_chain
    self.sample_pre = sample_pre
    self.sample_post = sample_post
  def __str__(self):
    return "SamplerChains(%s, %s, %s)" % (fl(self.request_chain), fl(self.sample_pre), fl(self.sample_post))

def default_put_chains(store):
  return SamplerChains(
    [store, M.SimpleSampling()],
    [M.PrepareInput()],
    [M.GetForcedInput()])

def default_get_chains(store, temperature = 1.0, endtoken = [], maxlength = None):
  return SamplerChains(
    [store, M.SimpleSampling()],
    [M.PrepareInput()],
    [M.ApplyTemperature(temperature), M.CalculateProbs(), M.SampleToken(), M.CheckEndingToken(endtoken), M.HardLengthLimit(maxlength)]
    )

class Sampler():
  def __init__(self, model):
    self.model = model

  def make_get_request(self, chains, key = ''):
    rq = SamplerRequest(key, chains)
    return rq

  def make_put_request(self, chains, sequence, key = ''):
    rq = SamplerRequest(key, chains)
    rq.forced_input = sequence
    return rq

  def single_step(self, samples):
    for s in samples:
      for mod in s.chains.sample_pre:
        mod.pre(s)
    nn_inputs = torch.LongTensor(util.ljoin([s.model_input_token for s in samples]))
    nn_states, nn_lengths = util.ljoinl([s.model_input_state for s in samples])
    nn_states = {k:v for k,v in enumerate(nn_states)}
    with torch.no_grad():
      nn_outputs, nn_outstates = self.model.forward_with_states(nn_inputs.unsqueeze(1), nn_states, out_device='cpu')
    nn_outputs_split = util.lsplitl(nn_outputs, nn_lengths)
    nn_outstates_split = util.lsplitl(nn_outstates, nn_lengths)
    for i,s in enumerate(samples):
      s.model_output_scores = nn_outputs_split[i]
      s.model_next_states = nn_outstates_split[i]
      for mod in s.chains.sample_post:
        mod.post(s)

  def run_requests(self, requests):
    for rq in requests:
      rq.run_inchain()
    samples = util.ljoin([rq.samples for rq in requests])
    while samples:
      self.single_step(samples)
      for s in [s for s in samples if s.finished]:
        samples.remove(s)
    for rq in requests:
      rq.run_outchain()
