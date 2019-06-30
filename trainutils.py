import time

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
