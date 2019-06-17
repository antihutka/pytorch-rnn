import sampling
import torch
import math

class DefaultStateStore:
  def __init__(self, model, default_token = 0):
    self.last_token = {}
    self.states = {}
    self.model = model
    self.default_token = default_token
  def forward(self, request):
    key = request.key
    if key in self.last_token:
      request.initial_token = self.last_token[key]
    else:
      request.initial_token = self.default_token # we'll do something better here later
    if key in self.states:
      request.initial_state = self.states[key]
    else:
      request.initial_state = None
    #print('loaded state for "%s"' % key, request.initial_state, request.initial_token)
  def backward(self, request):
    key = request.key
    #print('saving state for "%s"' % key, request.final_state, request.last_token)
    self.last_token[key] = request.last_token
    self.states[key] = request.final_state

class SimpleSampling:
  def forward(self, request):
    #print(request.__dict__)
    sample = sampling.Sample(request, request.chains, request.initial_state, request.initial_token)
    request.samples = [sample]
  def backward(self, request):
    [sample] = request.samples
    request.last_token = sample.sampled_sequence[-1]
    request.final_state = sample.states[-1]
    request.sampled_sequence = sample.sampled_sequence

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
    if (sample.request.forced_input.dim() > 1):
      assert sample.request.forced_input.dim() == 2
      assert sample.request.forced_input.size(0) == 1
      sample.request.forced_input = sample.request.forced_input[0]
    pos = sample.forced_pos if hasattr(sample, 'forced_pos') else 0
    sample.token_add(sample.request.forced_input[pos].item(), None, sample.model_next_states[0])
    sample.forced_pos = pos + 1
    if sample.forced_pos >= len(sample.request.forced_input):
      sample.finished = True

class PrintSampledString:
  def __init__(self, model):
    self.model = model
  def post(self, sample):
    print('=> %s' % (self.model.decode_string(sample.sampled_sequence).decode(errors='replace')))

class BlockBadWords:
  def __init__(self, model, badwords):
    self.model = model
    self.badwords = badwords
  def post(self, sample):
    decoded = self.model.decode_string(sample.sampled_sequence).decode(errors='replace').lower()
    if not hasattr(sample, 'bw_fails'):
      sample.bw_fails = {}
    if any((decoded.endswith(w.lower()) for w in self.badwords)):
      fails = sample.bw_fails.get(decoded, 0) + 1
      todel = max(1, math.floor(fails/3))
      sample.bw_fails[decoded] = fails
#      print('bad word detected, fails %d todel %d' % (fails, todel))
      sample.token_del(todel, True)
