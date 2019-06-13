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


class DefaultStateStore:
  def __init__(self, model):
    self.last_token = {}
    self.states = {}
    self.outputs = {}
    self.model = model
  def forward(self, request):
    key = request.key
    if key in self.outputs:
      request.initial_token = self.last_token[key]
    else:
      request.initial_token = 0 # we'll do something better here later
    if key in self.states:
      request.initial_state = self.states[key]
    else:
      request.initial_state = None
  def backward(self, request):
    key = request.key
    self.last_token[key] = request.last_token
    self.states[key] = request.final_state

class SimpleSampling:
  def forward(self, request):
    #print(request.__dict__)
    sample = Sample(request, request.chains, request.initial_state, request.initial_token)
    request.samples = [sample]
  def backward(self, request):
    [sample] = request.samples
    request.last_token = sample.sampled_sequence[-1]
    request.final_state = sample.states[-1]

class SamplerRequest:
  def __init__(self, key, chains):
    self.chains = chains
    self.key = key

class PrepareInput:
  def pre(self, sample):
    assert (not sample.finished)
    sample.model_input_token = [sample.input_tokens[-1]]
    sample.model_input_state = [sample.states[-1]]

class ApplyTemperature:
  def __init__(self, temperature):
    self.temperature = temperature
  def post(self, sample):
    sample.model_output_scores.div_(self.temperature)

class CalculateProbs:
  def post(self, sample):
    probs = sample.model_output_scores.double()[:, -1, :]
    probs.exp_()
    probs.div_(probs.sum(1, True))
    sample.model_output_probs = probs

class SampleToken:
  def post(self, sample):
    assert sample.model_output_probs.size(0) == 1
    probs = sample.model_output_probs
    token = torch.multinomial(probs, 1).item()
    sample.token_add(token, probs, sample.model_next_states[0])
    #print(sample.model_next_states[0])

class CheckEndingToken:
  def __init__(self, tokens):
    self.tokens = tokens
  def post(self, sample):
    if sample.sampled_sequence[-1] in self.tokens:
      sample.finished = True

class HardLengthLimit:
  def __init__(self, limit):
    self.limit = limit
  def post(self, sample):
    if self.limit and len(sample.sampled_sequence) >= self.limit:
      sample.finished = True

class GetForcedInput:
  def post(self, sample):
    pos = sample.forced_pos if hasattr(sample, 'forced_pos') else 0
    sample.token_add(sample.request.forced_input[pos], None, sample.model_next_states[0])
    sample.forced_pos = pos + 1
    if sample.forced_pos >= len(sample.request.forced_input):
      sample.finished = True

class PrintSampledString:
  def __init__(self, model):
    self.model = model
  def post(self, sample):
    print('=> %s' % (self.model.decode_string(sample.sampled_sequence).decode(errors='replace')))

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
    self.sampled_sequence += [token]
    self.input_tokens += [token]
    self.states += [state]
    self.probs += [probs]

class SamplerChains():
  def __init__(self, request_chain, sample_pre, sample_post):
    self.request_chain = request_chain
    self.sample_pre = sample_pre
    self.sample_post = sample_post

def default_put_chains(store):
  return SamplerChains(
    [store, SimpleSampling()],
    [PrepareInput()],
    [GetForcedInput()])

def default_get_chains(store, temperature = 1.0, endtoken = [], maxlength = None):
  return SamplerChains(
    [store, SimpleSampling()],
    [PrepareInput()],
    [ApplyTemperature(temperature), CalculateProbs(), SampleToken(), CheckEndingToken(endtoken), HardLengthLimit(maxlength)]
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

  def run_requests(self, requests):
    for rq in requests:
      for mod in rq.chains.request_chain:
        mod.forward(rq)
    samples = util.ljoin([rq.samples for rq in requests])
    while samples:
      for s in samples:
        for mod in s.chains.sample_pre:
          mod.pre(s)
      #print(samples[0].model_input_token)
      nn_inputs = torch.LongTensor(util.ljoin([s.model_input_token for s in samples]))
      nn_states, nn_lengths = util.ljoinl([s.model_input_state for s in samples])
      nn_states = {k:v for k,v in enumerate(nn_states)}
#      print("instate", nn_states[0])
      with torch.no_grad():
        nn_outputs, nn_outstates = self.model.forward_with_states(nn_inputs.unsqueeze(1), nn_states)
#      print("outstate", nn_outstates[0])
      nn_outputs_split = util.lsplitl(nn_outputs, nn_lengths)
      nn_outstates_split = util.lsplitl(nn_outstates, nn_lengths)
#      print(nn_outstates_split[0])
      for i,s in enumerate(samples):
        s.model_output_scores = nn_outputs_split[i]
        s.model_next_states = nn_outstates_split[i]
#        print(s.model_next_states)
        for mod in s.chains.sample_post:
          mod.post(s)
      for s in [s for s in samples if s.finished]:
        samples.remove(s)
    for rq in requests:
      for mod in reversed(rq.chains.request_chain):
        mod.backward(rq)
