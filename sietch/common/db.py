"""A SQLite connection shared by web handlers, the comms loop and the flusher thread.

sqlite3 objects are not safe to use from several threads at once (we hit "bad parameter or other API misuse"
under concurrent requests), so every statement runs under one lock and its rows are fetched inside it.
"""

import sqlite3
import threading


class Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class LockedDB:
    def __init__(self, path, synchronous="NORMAL"):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self.execute("PRAGMA journal_mode=WAL")
        self.execute(f"PRAGMA synchronous={synchronous}")

    def execute(self, sql, params=()):
        with self._lock:
            return Rows(self._conn.execute(sql, params).fetchall())

    def insert(self, sql, params=()):
        """Run an INSERT and return its rowid atomically (another thread can't slip in between)."""
        with self._lock:
            return self._conn.execute(sql, params).lastrowid

    def executescript(self, script):
        with self._lock:
            self._conn.executescript(script)

    def close(self):
        with self._lock:
            self._conn.close()
