import LanguageModel
import torch
import sampling
import argparse
import modules as M
import readline

readline.parse_and_bind('tab: complete')
readline.parse_and_bind('set editing-mode vi')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='models/test.json')
parser.add_argument('--maxlength', default=1024, type=int)
args = parser.parse_args()

model = LanguageModel.LanguageModel()
model.load_json(args.checkpoint)
model.eval()

sampler = sampling.Sampler(model)

stor = M.DefaultStateStore(model)
pc = sampling.default_put_chains(stor)
gc = sampling.default_get_chains(stor, endtoken = [model.token_to_idx[b'\n']], maxlength=args.maxlength)

#print(pc.__dict__)
#print(gc.__dict__)

gc.sample_post += [M.PrintSampledString(model)]

while True:
    line = input('>')
    if line != '':
      sampler.run_requests([sampler.make_put_request(pc, model.encode_string(line + '\n'))])
    sampler.run_requests([sampler.make_get_request(gc)])
