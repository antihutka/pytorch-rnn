import torch

class Sampler():
  def __init__(self, model):
    self.model = model
    self.queued_inputs = {}
    self.outputs = {}
    self.modifiers = []

  def calculate_probs(self, x):
    x = x[-1, :]
    x = x.double()
    x.exp_()
    x.div_(x.sum(0).unsqueeze(0))
    return x

  def sample_from_probs(self, p):
    return torch.multinomial(p, 1)

  def push_single(self, value, key = ''):
    evalue = self.model.encode_string(value)
    if key not in self.queued_inputs:
      self.queued_inputs[key] = [evalue]
    else:
      self.queued_inputs[key] += [evalue]

  def push(self, values):
    for k,v in values.items():
      self.push_single(v, k)

  def has_queued_input(self, key):
    return key in self.queued_inputs

  def pull_queued_input(self, key):
    t = self.queued_inputs[key][0]
    if t.size(1) > 1:
      self.queued_inputs[key][0] = t[:, 1:]
    elif len(self.queued_inputs[key]) > 1:
      del self.queued_inputs[key][0]
    else:
      del self.queued_inputs[key]
    return t[:, 0].item()

  def next_input_for_key(self, key):
    if self.has_queued_input(key):
      return self.pull_queued_input(key)
    else:
      return self.sample_key(key)

  def sample_key(self, key):
    scores = self.outputs.get(key, torch.ones(1, len(self.model.idx_to_token)))
    for mod in self.modifiers:
      scores = mod.modify_scores(key, scores)
    probs = self.calculate_probs(scores)
    for mod in self.modifiers:
      probs = mod.modify_probs(key, probs)
    selected_token = self.sample_from_probs(probs).item()
    for mod in self.modifiers:
      mod.token_sampled(key, selected_token, probs.squeeze())
    return selected_token

  def sample(self, keys = ['']):
    for k in keys:
      for mod in self.modifiers:
        mod.sample_start(k)
    while keys:
      inputs = torch.LongTensor([ [self.next_input_for_key(key)] for key in keys ])
      with torch.no_grad():
        print('calling forward with ', inputs.size())
        outputs = self.model.forward(inputs)
      for idx, key in enumerate(keys):
        self.outputs[key] = outputs[idx]
      to_stop = [k for k in keys if any([mod.should_stop(k) for mod in self.modifiers])]
      for k in to_stop:
        keys.remove(k)
        for mod in self.modifiers:
          mod.sample_finish(k)

class SamplerModifier():
  def modify_probs(self, key, probs):
    return probs
  def modify_scores(self, key, scores):
    return scores
  def token_sampled(self, key, token_id, probs):
    pass
  def should_stop(self, key):
    return False
  def sample_start(self, key):
    pass
  def sample_finish(self, key):
    pass

class TemperatureModifier(SamplerModifier):
  def __init__(self, temperature):
    super(TemperatureModifier, self).__init__()
    self.temperature = temperature

  def modify_scores(self, key, scores):
    return scores.div_(self.temperature)

class HardLengthLimit(SamplerModifier):
  def __init__(self, max_length):
    super(HardLengthLimit, self).__init__()
    self.current_length = {}
    self.max_length = max_length
  def token_sampled(self, key, token_id, probs):
    self.current_length[key] += 1
  def sample_start(self, key):
    self.current_length[key] = 0
  def should_stop(self, key):
    return self.current_length[key] >= self.max_length

class StopOnToken(SamplerModifier):
  def __init__(self, tokens):
    super(StopOnToken, self).__init__()
    self.tokens = tokens
    self.should_stop_now = False
  def token_sampled(self, key, token_id, probs):
    self.should_stop_now = token_id in self.tokens
  def should_stop(self, key):
    return self.should_stop_now

