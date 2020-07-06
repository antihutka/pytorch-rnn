import MySQLdb
import pickle
import threading
import logging
import random

logger = logging.getLogger(__name__)

create_table = '''
CREATE TABLE IF NOT EXISTS `state_storage` (
  `modelid` INT NOT NULL,
  `key` VARCHAR(128) NOT NULL,
  `state` LONGBLOB NOT NULL,
  `last_token` INT NOT NULL,
  `modified_on` TIMESTAMP on update CURRENT_TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`modelid`, `key`),
  INDEX (`modified_on`)); 
'''

class MySQLStore:
  def __init__(self, model, dbhost, dbname, dbuser, dbpass, default_token=0, commit_every = 256, max_cache = 1024, modelid=0):
    self.model = model
    self.default_token = default_token
    self.dbhost = dbhost
    self.dbname = dbname
    self.dbuser = dbuser
    self.dbpass = dbpass
    self.commit_every = commit_every
    self.modelid = modelid
    self.max_cache = max_cache
    self.cached = {}
    self.dirty = set()
    self.lock = threading.RLock()
    self.writes = 0
    c = self.opendb()
    c.cursor().execute(create_table)
    c.commit()
    c.close()

  def opendb(self):
    conn = MySQLdb.connect(host=self.dbhost, database=self.dbname, user=self.dbuser, password=self.dbpass)
    return conn

  def get_state(self, key):
    with self.lock:
      if key in self.cached:
        return self.cached[key]
      con = self.opendb()
      try:
        cur = con.cursor()
        cur.execute('SELECT `state`, `last_token` FROM `state_storage` WHERE `modelid` = %s AND `key` = %s', (self.modelid, key))
        self.check_cache_size()
        return cur.fetchone()
      finally:
        con.close()

  def check_cache_size(self):
    #logger.info("Current cache size %d", len(self.cached))
    if len(self.cached) > self.max_cache:
      cleankeys = [x for x in self.cached.keys() if (x not in self.dirty)]
      random.shuffle(cleankeys)
      for k in cleankeys[:len(cleankeys)//8]:
        del self.cached[k]

  def write_state(self, key, state, token):
    with self.lock:
      self.cached[key] = (state, token)
      self.dirty.add(key)
      self.writes += 1
      if self.writes > self.commit_every:
        self.commit()

  def forward(self, request):
    r = self.get_state(request.key)
    if r:
      (state, token) = r
    else:
      r = self.get_state('_default')
      if r:
        logger.info("loading default state for %s", request.key)
        (state, token) = r
      else:
        logger.info("loading empty state for %s", request.key)
        (state, token) = (None, self.default_token)
    request.initial_state = pickle.loads(state) if state else None
    request.initial_token = token

  def backward(self, request):
    #for k,v in request.final_state.items():
    #  print('state', k, v, v.size())
    #print('serialized size:', len(pickle.dumps(request.final_state)))
    self.write_state(request.key, pickle.dumps(request.final_state), request.last_token)

  def commit(self):
    with self.lock:
      logger.info("Commiting %d states", len(self.dirty))
      if len(self.dirty) == 0:
        return
      try:
        dirty_states = [(self.modelid, x) + self.cached[x] for x in self.dirty]
        con = self.opendb()
        try:
          cur = con.cursor()
          cur.executemany("REPLACE INTO state_storage (`modelid`, `key`, `state`, `last_token`) VALUES (%s, %s, %s, %s)", dirty_states)
          con.commit()
          self.writes = 0
          self.dirty.clear()
        finally:
          con.close()
        logger.info("Commited")
      except Exception:
        logger.exception("Error writing states")
