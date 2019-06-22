import sqlite3
import pickle
import threading
import logging

logger = logging.getLogger(__name__)


class SQLiteStateStore:
  def __init__(self, model, db_path, default_token=0, commit_every = 256):
    self.t = threading.local()
    self.db_path = db_path
    self.opendb()
    self.model = model
    self.default_token = default_token
    self.t.conn.execute("CREATE TABLE IF NOT EXISTS states (key TEXT PRIMARY KEY, last_token INTEGER, state BINARY)")
    self.t.conn.commit()
    self.writes = 0
    self.commit_every = commit_every

  def opendb(self):
    if hasattr(self.t, 'conn'):
      return
    self.t.conn = sqlite3.connect(self.db_path)
    self.t.conn.execute("PRAGMA journal_mode=WAL")

  def forward(self, request):
    self.opendb()
    key = request.key
    r = self.t.conn.execute("SELECT last_token, state FROM states WHERE key = ?", (key,)).fetchall()
    if r:
      [(token, state)] = r
      request.initial_token = token
      request.initial_state = pickle.loads(state)
    else:
      r = self.t.conn.execute("SELECT last_token, state FROM states WHERE key = '_default'").fetchall()
      if r:
        logger.info("loading default state for %s" % key)
        [(token, state)] = r
        request.initial_token = token
        request.initial_state = pickle.loads(state)
      else:
        logger.info("loading empty state for %s" % key)
        request.initial_token = self.default_token
        request.initial_state = None

  def backward(self, request):
    self.opendb()
    self.t.conn.execute("INSERT OR REPLACE INTO states (key, last_token, state) VALUES (?,?,?)", (request.key, request.last_token, pickle.dumps(request.final_state)))
    self.writes += 1
    if self.writes > self.commit_every or getattr(request, 'force_commit', False):
      self.commit()

  def commit(self):
    if self.writes == 0:
      return
    logger.info("Commiting")
    self.t.conn.commit()
    self.writes = 0
    logger.info("Commit done")

  def __str__(self):
    return "SQLiteStateStore()"
