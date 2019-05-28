import LanguageModel
import torch
import sampling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='models/test.json')
args = parser.parse_args()

model = LanguageModel.LanguageModel()
model.load_json(args.checkpoint)
model.eval()

sampler = sampling.Sampler(model)

class TestModifier(sampling.SamplerModifier):
  def __init__(self, model):
    super(TestModifier, self).__init__()
    self.current_str = b''
    self.model = model
  def token_sampled(self, key, token_id, probs):
    print('got token %d prob %.2f' % (token_id, probs[token_id]))
    self.current_str += model.idx_to_token[token_id]

tm = TestModifier(model)

sampler.modifiers += [
  sampling.TemperatureModifier(0.8),
  sampling.HardLengthLimit(50),
  tm,
  sampling.StopOnToken([model.token_to_idx[b'\n']])]

sampler.push_single('Hello!')
sampler.sample()
print('generated output: %s' % tm.current_str.decode(errors='replace'))

