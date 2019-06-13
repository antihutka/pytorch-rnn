def ljoin(l):
  return [x for y in l for x in y]

def ljoinl(l):
  joined = []
  lengths = []
  for e in l:
    joined += e
    lengths += [len(e)]
  return joined, lengths

def lsplitl(lst, lengths):
  pos = 0
  out = []
  for length in lengths:
    out += [lst[pos:pos+length]]
    pos += length
  assert pos == len(lst)
  return out
