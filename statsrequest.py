import time
import sampling
import logging

def decodeseq(model, seq):
  try:
    s = model.decode_string(seq).decode('utf8', errors='backslashreplace')
  except Exception as e:
    logging.exception("Failed to decode sequence")
    s = '??? %s\n ' % (str(seq))
  return repr(s)

def req2str(model, reqs):
  out = ''
  for req in reqs:
    out += '---- %s\n' % req.key
    for (k,v) in req.__dict__.items():
      if k == 'forced_input':
        out += '  forced_input : %s\n' % decodeseq(model, v)
      elif k not in ['initial_state', 'samples', 'key', 'on_finish', 'chains']:
        out += '  %s : %s\n' % (k,v)
    for (i,s) in enumerate(req.samples):
      out += '  ---- sample %d\n' % i
      for (k,v) in s.__dict__.items():
        if k in ['model_output_scores', 'states', 'model_output_probs', 'probs', 'model_next_states']:
          out += '    %s : [%d values]\n' % (k, len(v))
        elif k in ['sampled_sequence', 'input_tokens']:
          out += '    %s : %s\n' % (k,decodeseq(model, v))
        elif k not in ['model_input_token', 'model_input_state']:
          out += '    %s : %s\n' % (k,v)
  return out
class StatsRequestModule():
  def __init__(self, sampler):
    self.sampler = sampler
  def forward(self, request):
    request.start_time = time.clock()
  def backward(self, request):
    request.end_time = time.clock()
    request.elapsed = request.end_time - request.start_time
    request.requestinfo = req2str(self.sampler.sampler.model, self.sampler.requests)

class StatsRequest(sampling.SamplerRequest):
  def __init__(self, sampler):
    self.chains = sampling.SamplerChains([StatsRequestModule(sampler)], [], [])
    self.key = None
    self.samples = []

