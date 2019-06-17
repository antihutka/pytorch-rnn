import LanguageModel
import torch
import sampling
import argparse
import modules as M
import readline
import os
import pickle

readline.parse_and_bind('tab: complete')
readline.parse_and_bind('set editing-mode vi')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='models/test.json')
parser.add_argument('--maxlength', default=1024, type=int)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--savedir', default='')
args = parser.parse_args()

model = LanguageModel.LanguageModel()
model.load_json(args.checkpoint)
model.eval()

sampler = sampling.Sampler(model)

stor = M.DefaultStateStore(model, default_token = model.token_to_idx[b'\n'])
pc = sampling.default_put_chains(stor)
gc = sampling.default_get_chains(stor, endtoken = [model.token_to_idx[b'\n']], maxlength=args.maxlength, temperature = args.temperature)

badword_mod = M.BlockBadWords(model, [])

path_bw = args.savedir + '/badwords'
if args.savedir and os.path.exists(path_bw):
  badword_mod.badwords = pickle.load(open(path_bw, 'rb'))

gc.sample_post += [M.PrintSampledString(model), badword_mod]

def in_msg(line):
  if line != '':
    sampler.run_requests([sampler.make_put_request(pc, model.encode_string(line + '\n'))])
  sampler.run_requests([sampler.make_get_request(gc)])

def in_cmd(line):
  if ' ' not in line:
    line += ' '
  [cmd, arg] = line.split(' ', 1)
  if (cmd == 'abw' and arg != ''):
    badword_mod.badwords.append(arg)
    if args.savedir:
      with open(path_bw, 'wb') as f:
        pickle.dump(badword_mod.badwords, f)
  elif (cmd == 'dbw' and arg != ''):
    badword_mod.badwords.remove(arg)
    if args.savedir:
      with open(path_bw, 'wb') as f:
        pickle.dump(badword_mod.badwords, f)
  elif (cmd == 'pbw'):
    print('current bad words: ', badword_mod.badwords)
  else:
    print('unknown command %s' % cmd)

while True:
    line = input('>')
    if line.startswith('//'):
      in_msg(line[1:])
    elif line.startswith('/'):
      in_cmd(line[1:])
    else:
      in_msg(line)

