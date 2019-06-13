import sampling
import torch

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
    sample = sampling.Sample(request, request.chains, request.initial_state, request.initial_token)
    request.samples = [sample]
  def backward(self, request):
    [sample] = request.samples
    request.last_token = sample.sampled_sequence[-1]
    request.final_state = sample.states[-1]

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
