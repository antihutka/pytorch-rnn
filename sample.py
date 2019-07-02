import LanguageModel
import torch
import sampling
import argparse
import modules as M

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='models/test.json')
args = parser.parse_args()

model = LanguageModel.LanguageModel()
model.load_json(args.checkpoint)
model.eval()


sampler = sampling.Sampler(model)


stor = M.DefaultStateStore(model)
pc = sampling.default_put_chains(stor)
gc = sampling.default_get_chains(stor, endtoken=[model.token_to_idx[b'\n']])

#print(pc.__dict__)
#print(gc.__dict__)

#gc.sample_post += [M.PrintSampledString(model)]

sampler.run_requests([sampler.make_put_request(pc, model.encode_string('Hello!\n'))])
print('ok!')
while True:
  req = sampler.make_get_request(gc)
  sampler.run_requests([req])
  print(model.decode_string(req.sampled_sequence).decode(errors='backslashreplace'), end="")
