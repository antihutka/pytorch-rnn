import sampling
import queue
from threading import Thread, Event
import logging

class SamplerServer():
  def __init__(self, model):
    self.sampler = sampling.Sampler(model)
    self.queue = queue.Queue()
    self.thread = Thread(target=self.threadmain_h)
    self.thread.daemon = True
    self.stopped = True
    self.thread.start()
  def stop(self):
    self.queue.put(None)
    self.queue.join()
    self.thread.join()
    self.stopped = True
  def __del__(self):
    if not self.stopped:
      self.stop()

  def threadmain_h(self):
    while True:
      try:
        self.threadmain()
      except Exception:
        logging.exception("Exception in server thread")
        for r in self.requests:
          r.on_finish()

  def threadmain(self):
    requests = []
    samples = []
    self.requests = requests
    stop = False
    while True:
      while (not requests) or (not self.queue.empty()):
        r = self.queue.get(True)
        if (r is None):
          self.queue.task_done()
          stop = True
          break
        r.run_inchain()
        requests.append(r)
        samples.extend(r.samples)
      if (not requests) and stop:
        return
      if samples:
        self.sampler.single_step(samples)
        for s in [s for s in samples if s.finished]:
          samples.remove(s)
      for r in requests:
        if all([rs.finished for rs in r.samples]):
          requests.remove(r)
          r.run_outchain()
          r.finished = True
          r.on_finish()
          self.queue.task_done()

  def run_request_sync(self, request):
    evt = Event()
    request.on_finish = lambda: evt.set()
    self.queue.put(request)
    evt.wait()
    assert request.finished
    return request

  def run_request(self, request, on_finish):
    request.on_finish = on_finish
    self.queue.put(request)
