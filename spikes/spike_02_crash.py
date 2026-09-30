"""Spike 02: isolate Edge crash durability. Child process writes then os._exit()s; parent reloads.

Variants: no flush / flush / sleep before exit / N points / reload twice.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r"""
import os, sys, time, uuid
sys.path.insert(0, {here!r})
import spike_01_engine as s
from qdrant_edge import EdgeShard, Point, UpdateOperation
path = {path!r}
os.makedirs(path, exist_ok=True)
sh = EdgeShard.create(path, s.config())
for i in range({n}):
    sh.update(UpdateOperation.upsert_points([Point(str(uuid.UUID(int=i + 1)), {{s.DENSE: s.rvec(i)}}, {{'i': i}})]))
mode = {mode!r}
if mode == 'flush':
    sh.flush()
elif mode == 'sleep':
    time.sleep(2)
os._exit(0)
"""


def wal_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        if "wal" in root.lower():
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def run(mode, n):
    sys.path.insert(0, HERE)
    from qdrant_edge import EdgeShard

    tmp = tempfile.mkdtemp(prefix="edge_crash_")
    path = os.path.join(tmp, "s")
    try:
        subprocess.run([sys.executable, "-c", CHILD.format(here=HERE, path=path, n=n, mode=mode)], check=True)
        wal = wal_bytes(path)
        sh = EdgeShard.load(path)
        got = sh.retrieve([str(uuid.UUID(int=i + 1)) for i in range(n)], with_payload=False, with_vector=False)
        sh.close()
        sh = EdgeShard.load(path)  # reload after clean close
        got2 = sh.retrieve([str(uuid.UUID(int=i + 1)) for i in range(n)], with_payload=False, with_vector=False)
        sh.close()
        print(f"mode={mode:6} n={n:5}  wal_bytes_after_crash={wal:8}  recovered={len(got)}/{n}  after_2nd_reload={len(got2)}/{n}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for mode in ("none", "sleep", "flush"):
        for n in (1, 50, 2000):
            run(mode, n)
