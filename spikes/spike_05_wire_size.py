"""Spike 05: on-the-wire size of snapshots. Does the server compress (Accept-Encoding)? How small is zstd/gzip?"""

import gzip
import io
import os
import sys
import tempfile
import time
import uuid

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spike_01_engine as s  # noqa: E402
from qdrant_edge import EdgeShard  # noqa: E402

URL, C = "http://localhost:6333", "fleet"


def wire(method, path, enc, **kw):
    r = requests.request(method, URL + path, headers={"Accept-Encoding": enc}, stream=True, timeout=120, **kw)
    r.raise_for_status()
    raw = r.raw.read(decode_content=False)
    return len(raw), r.headers.get("content-encoding"), raw


def main():
    sid = 1  # zone_a from spike 04 (collection left in place)
    n_id, ce_id, raw = wire("GET", f"/collections/{C}/shards/{sid}/snapshot", "identity")
    n_gz, ce_gz, _ = wire("GET", f"/collections/{C}/shards/{sid}/snapshot", "gzip, deflate, br, zstd")
    print(f"full: identity={n_id/1e6:.1f} MB | server with Accept-Encoding gzip/zstd -> {n_gz/1e6:.1f} MB (content-encoding={ce_gz})")
    t0 = time.perf_counter()
    gz = gzip.compress(raw, compresslevel=6)
    print(f"full: client-side gzip -6 -> {len(gz)/1e6:.2f} MB in {time.perf_counter()-t0:.1f}s ({len(raw)/max(len(gz),1):.0f}x)")

    # partial after a single payload change, from a fresh edge copy of the current shard
    tmp = tempfile.mkdtemp(prefix="edge_w_")
    snap = os.path.join(tmp, "f.snapshot")
    open(snap, "wb").write(raw)
    edge_dir = os.path.join(tmp, "m")
    EdgeShard.unpack_snapshot(snap, edge_dir)
    edge = EdgeShard.load(edge_dir)
    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "zone_a-1002"))
    requests.post(URL + f"/collections/{C}/points/payload?wait=true",
                  json={"payload": {"touched2": True}, "points": [pid], "shard_key": "zone_a"}).raise_for_status()
    time.sleep(2)
    m = edge.snapshot_manifest()
    n_p, _, praw = wire("POST", f"/collections/{C}/shards/{sid}/snapshot/partial/create", "identity", json=m)
    pgz = gzip.compress(praw, compresslevel=6)
    print(f"partial (1 payload change): raw={n_p/1e6:.1f} MB, gzip={len(pgz)/1e3:.0f} KB")
    edge.close()


if __name__ == "__main__":
    main()
