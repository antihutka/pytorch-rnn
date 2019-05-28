import LanguageModel
import argparse
import torch
import sys
import time
import resource

torch.no_grad()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='models/test.json')
args = parser.parse_args()

model = LanguageModel.LanguageModel()
model.load_json(args.checkpoint)
model.eval()

encoded = model.encode_string("Hello!")
#encoded = encoded.unsqueeze(0)
print(encoded)

with torch.no_grad():
  out = model.forward(encoded)[:, -1]
print(out.pow(2).sum())
#exit(0)

probs = out.double().exp().squeeze()
probs.div_(probs.sum())
#print(probs)
#for i,p in enumerate(probs):
#  if p.item() > 0.01:
#    print(model.idx_to_token[i], "%.2f" % p.item())

start = time.time()


inp = torch.LongTensor(1,1)

with torch.no_grad():
  for i in range(0,200):
    probs = out.double().div(1).exp().squeeze()
    probs.div_(probs.sum())
    next_char_idx = torch.multinomial(probs, 1).item()
#  sys.stdout.write(model.idx_to_token[next_char_idx].decode(errors='ignore'))
#  sys.stdout.flush()
    inp[0,0] = next_char_idx
    out = model.forward(inp)[:, -1]
#    if i % 100 == 0:
#      print("memory @ %d: %.2f" % (i, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


end = time.time()

print("time: %.2fs" % (end - start))
print("memory: %.2f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
