import time
import torch

class TrainLog:
  def __init__(self):
    self.time = []
  def add(iteration, time):
    self.time.append(time)

class Timer:
  def __init__(self):
    self.total = 0
    self.count = 0
  def __enter__(self):
    self.start()
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()
  def reset(self):
    self.total = 0
    self.count = 0
  def start(self):
    self.starttime = time.time()
  def stop(self):
    self.endtime = time.time()
    self.last = self.endtime - self.starttime
    self.total += self.last
    self.count += 1
  def average(self):
    return self.total / self.count

class Average:
  def __init__(self, cnt):
    self.valid_cnt = 0
    self.wpos = 0
    self.cnt = cnt
    self.values = torch.zeros(cnt)
    self.sum = 0
  def add_value(self, value):
    self.sum = self.sum - self.values[self.wpos] + value
    self.values[self.wpos] = value
    self.wpos = (self.wpos + 1) % self.cnt
    if self.valid_cnt < self.cnt:
      self.valid_cnt += 1
  def avg(self):
    return self.sum / self.valid_cnt

class ValueHistory:
  def __init__(self, name):
    self.name = name
    self.values = []
    self.steps = []
  def add_value(self, i, value):
    self.steps.append(i)
    self.values.append(value)
  def format(self):
    s = "%s" % self.name
    if self.values:
      s += "[%d]=%.4f" % (self.steps[-1], self.values[-1])
    else:
      s += " ???"
    if len(self.values) > 1:
      s += " diff=%.4f" % (self.values[-1] - self.values[-2])
      s += " best=%.4f at %d" % (min(self.values), self.steps[self.values.index(min(self.values))])
    return s
