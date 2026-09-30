"""Spike 03: (a) are writes after an earlier flush lost on hard exit? (b) flush latency vs batch size."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spike_01_engine as s  # noqa: E402
from qdrant_edge import EdgeShard, Point, UpdateOperation  # noqa: E402

CHILD = r"""
import os, sys, uuid
sys.path.insert(0, {here!r})
import spike_01_engine as s
from qdrant_edge import EdgeShard, Point, UpdateOperation
sh = EdgeShard.create({path!r}, s.config())
up = lambda i: sh.update(UpdateOperation.upsert_points([Point(str(uuid.UUID(int=i)), {{s.DENSE: s.rvec(i)}}, {{'i': i}})]))
for i in range(1, 11): up(i)
sh.flush()
for i in range(11, 21): up(i)
os._exit(0)
"""


def after_flush_then_crash():
    tmp = tempfile.mkdtemp(prefix="edge_f_")
    path = os.path.join(tmp, "s")
    os.makedirs(path)
    try:
        subprocess.run([sys.executable, "-c", CHILD.format(here=HERE, path=path)], check=True)
        sh = EdgeShard.load(path)
        a = len(sh.retrieve([str(uuid.UUID(int=i)) for i in range(1, 11)], with_payload=False, with_vector=False))
        b = len(sh.retrieve([str(uuid.UUID(int=i)) for i in range(11, 21)], with_payload=False, with_vector=False))
        sh.close()
        print(f"flushed batch recovered {a}/10, post-flush unflushed batch recovered {b}/10")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def flush_cost():
    tmp = tempfile.mkdtemp(prefix="edge_fc_")
    try:
        sh, _ = s.new_shard(tmp)
        s.seed_points(sh)
        n = 0
        for batch in (1, 10, 100, 1000):
            ts = []
            for _ in range(5):
                pts = [Point(str(uuid.uuid4()), {s.DENSE: s.rvec(n + j)}, {"j": j}) for j in range(batch)]
                n += batch
                t0 = time.perf_counter()
                sh.update(UpdateOperation.upsert_points(pts))
                t1 = time.perf_counter()
                sh.flush()
                t2 = time.perf_counter()
                ts.append(((t1 - t0) * 1000, (t2 - t1) * 1000))
            up = sorted(t[0] for t in ts)[2]
            fl = sorted(t[1] for t in ts)[2]
            print(f"batch={batch:5}  median upsert={up:7.2f} ms  median flush={fl:7.2f} ms  (total points {n})")
        sh.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    after_flush_then_crash()
    flush_cost()
