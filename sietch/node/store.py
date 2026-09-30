"""Durable on-device memory: two Qdrant Edge shards plus a SQLite write-ahead log of our own.

Why our own log: qdrant-edge-py 0.8.0 on Windows loses every write since the last flush() on a hard exit, even
though its docs say returned updates survive a crash (spike_02/03). flush() costs ~16-23 ms regardless of size,
so every write is appended to SQLite first (synchronous=FULL), applied to the shard, and a background thread
group-commits: flush the shards, then drop log entries up to the flushed sequence. On start, entries after the
last checkpoint are replayed (upserts/sets/deletes are idempotent).

Shards:
  memory - this node's observations (dense "img" + sparse "tags"), mutable
  fleet  - what ground control already holds from the whole fleet (dense only), used for redundancy/novelty
"""

import json
import os
import threading
import time

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeSparseVectorParams,
    EdgeVectorParams,
    Modifier,
    PayloadSchemaType,
    Point,
    ScrollRequest,
    SparseVector,
    UpdateOperation,
)

from sietch.common.db import LockedDB

MEMORY_INDEXES = {
    "state": PayloadSchemaType.Keyword,  # new | sent | thumb | full | held
    "node": PayloadSchemaType.Keyword,
    "requested_full": PayloadSchemaType.Integer,  # 0/1: boolean MatchValue never matches on Edge 0.8.0 Python
    "superseded": PayloadSchemaType.Integer,  # 0/1, same reason
    "kind": PayloadSchemaType.Keyword,  # observation | insight | decision | answer | note
    "importance": PayloadSchemaType.Float,
    "captured_at": PayloadSchemaType.Datetime,
    "novelty": PayloadSchemaType.Float,
    "influence": PayloadSchemaType.Float,
    "bytes_next": PayloadSchemaType.Float,
    "geo": PayloadSchemaType.Geo,  # where it was seen: spatial recall ("what's around here?")
}


def _open(path, config, indexes):
    if os.path.exists(os.path.join(path, "edge_config.json")):
        return EdgeShard.load(path)
    os.makedirs(path, exist_ok=True)
    shard = EdgeShard.create(path, config)
    for field, schema in indexes.items():
        shard.update(UpdateOperation.create_field_index(field, schema))
    shard.flush()
    return shard


class Store:
    def __init__(self, data_dir, dim):
        self.dir = data_dir
        self.img_dir = os.path.join(data_dir, "img")
        os.makedirs(self.img_dir, exist_ok=True)
        self._lock = threading.RLock()
        self.db = LockedDB(os.path.join(data_dir, "node.db"), synchronous="FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS ops (seq INTEGER PRIMARY KEY AUTOINCREMENT, shard TEXT, op TEXT);
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, text TEXT);
            CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, window INTEGER,
                item TEXT, rep TEXT, bytes INTEGER);
            """
        )
        self.shards = {
            "memory": _open(
                os.path.join(data_dir, "memory"),
                EdgeConfig(
                    vectors={"img": EdgeVectorParams(size=dim, distance=Distance.Cosine)},
                    sparse_vectors={"tags": EdgeSparseVectorParams(modifier=Modifier.Idf)},
                ),
                MEMORY_INDEXES,
            ),
            "fleet": _open(
                os.path.join(data_dir, "fleet"),
                EdgeConfig(vectors={"img": EdgeVectorParams(size=dim, distance=Distance.Cosine)}),
                {"node": PayloadSchemaType.Keyword},
            ),
        }
        self.replayed = self._replay()
        self._dirty = False
        self._stop = threading.Event()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()

    # ------------------------------------------------------------ write path

    def _log_and_apply(self, shard, op):
        with self._lock:
            self.db.execute("INSERT INTO ops (shard, op) VALUES (?, ?)", (shard, json.dumps(op)))
            self._apply(shard, op)
            self._dirty = True

    def _apply(self, shard, op):
        sh = self.shards[shard]
        kind = op["kind"]
        if kind == "upsert":
            vec = {"img": op["vec"]}
            if op.get("sparse"):
                vec["tags"] = SparseVector(op["sparse"]["i"], op["sparse"]["v"])
            sh.update(UpdateOperation.upsert_points([Point(op["id"], vec, op["payload"])]))
        elif kind == "set":
            sh.update(UpdateOperation.set_payload(op["ids"], op["payload"]))
        elif kind == "delete":
            sh.update(UpdateOperation.delete_points(op["ids"]))
        else:
            raise ValueError(kind)

    def upsert(self, pid, vec, payload, sparse=None, shard="memory"):
        op = {"kind": "upsert", "id": pid, "vec": [float(x) for x in vec], "payload": payload}
        if sparse is not None:
            op["sparse"] = {"i": list(sparse.indices), "v": list(sparse.values)}
        self._log_and_apply(shard, op)

    def set_payload(self, ids, payload, shard="memory"):
        if ids:
            self._log_and_apply(shard, {"kind": "set", "ids": list(ids), "payload": payload})

    def delete(self, ids, shard="memory"):
        if ids:
            self._log_and_apply(shard, {"kind": "delete", "ids": list(ids)})

    # ------------------------------------------------------------ durability

    def _replay(self):
        after = int(self.meta("flushed_seq", "0"))
        rows = self.db.execute("SELECT seq, shard, op FROM ops WHERE seq > ? ORDER BY seq", (after,)).fetchall()
        with self._lock:
            for _, shard, op in rows:
                self._apply(shard, json.loads(op))
        if rows:
            self.checkpoint()
        return len(rows)

    def checkpoint(self):
        with self._lock:
            seq = self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM ops").fetchone()[0]
            for sh in self.shards.values():
                sh.flush()
            self.set_meta("flushed_seq", str(seq))
            self.db.execute("DELETE FROM ops WHERE seq <= ?", (seq,))
            self._dirty = False

    def _flush_loop(self):
        while not self._stop.wait(0.25):
            if self._dirty:
                self.checkpoint()

    def close(self):
        self._stop.set()
        self._flusher.join(timeout=2)
        self.checkpoint()
        for sh in self.shards.values():
            sh.close()
        self.db.close()

    # ------------------------------------------------------------ read path

    def query(self, request, shard="memory"):
        with self._lock:
            return self.shards[shard].query(request)

    def get(self, ids, shard="memory", with_vector=False):
        with self._lock:
            return self.shards[shard].retrieve(list(ids), with_payload=True, with_vector=with_vector)

    def scroll(self, flt=None, shard="memory", with_vector=False):
        out, offset = [], None
        with self._lock:
            while True:
                recs, offset = self.shards[shard].scroll(
                    ScrollRequest(limit=256, offset=offset, filter=flt, with_payload=True, with_vector=with_vector)
                )
                out += recs
                if offset is None:
                    return out

    def count(self, shard="memory"):
        return len(self.scroll(shard=shard))

    # ------------------------------------------------------------ meta, activity, byte ledger

    def meta(self, k, default=None):
        row = self.db.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
        return row[0] if row else default

    def set_meta(self, k, v):
        self.db.execute("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (k, v))

    def event(self, kind, text):
        self.db.execute("INSERT INTO events (ts, kind, text) VALUES (?, ?, ?)", (time.time(), kind, text))

    def events(self, limit=50):
        rows = self.db.execute("SELECT ts, kind, text FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": ts, "kind": k, "text": t} for ts, k, t in rows]

    def record_sent(self, window, item, rep, nbytes):
        self.db.execute("INSERT INTO ledger (ts, window, item, rep, bytes) VALUES (?, ?, ?, ?, ?)",
                        (time.time(), window, item, rep, nbytes))

    def bytes_sent(self):
        b, n = self.db.execute("SELECT COALESCE(SUM(bytes), 0), COUNT(*) FROM ledger").fetchone()
        return {"bytes": b, "items": n}
