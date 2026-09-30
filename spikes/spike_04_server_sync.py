"""Spike 04: Qdrant Server 1.19.1 <-> Edge 0.8.0 snapshot sync.

Checks: custom sharding per zone -> per-shard edge init; partial snapshot carries inserts,
payload updates and DELETES; delta size vs full; what happens to edge-local writes in a mirror
shard when a partial is applied.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import uuid

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spike_01_engine as s  # noqa: E402
from qdrant_edge import EdgeShard, Point, Query, QueryRequest, UpdateOperation  # noqa: E402

URL = "http://localhost:6333"
C = "fleet"


def req(method, path, **kw):
    r = requests.request(method, URL + path, timeout=60, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
    return r


def uid(zone, i):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{zone}-{i}"))


def upsert(zone, rng, extra=None):
    pts = []
    for i in rng:
        text = f"{zone} note {i} " + ("E42 inverter" if i % 7 == 0 else "battery link")
        sp = s.BM25.embed_document(text)
        pts.append({
            "id": uid(zone, i),
            "vector": {s.DENSE: s.rvec(i), s.SPARSE: {"indices": sp.indices, "values": sp.values}},
            "payload": {"zone": zone, "i": i, "text": text, **(extra or {})},
        })
    req("PUT", f"/collections/{C}/points?wait=true", json={"points": pts, "shard_key": zone})


def shard_id_for(zone):
    info = req("GET", f"/collections/{C}/cluster").json()["result"]
    for sh in info["local_shards"]:
        if sh.get("shard_key") == zone:
            return sh["shard_id"]
    raise RuntimeError(f"no shard for {zone}: {info}")


def download_full(shard_id, dest):
    r = req("GET", f"/collections/{C}/shards/{shard_id}/snapshot", stream=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return os.path.getsize(dest)


def download_partial(shard_id, manifest, dest):
    r = req("POST", f"/collections/{C}/shards/{shard_id}/snapshot/partial/create", json=manifest, stream=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return os.path.getsize(dest), r.headers.get("content-type")


def breakdown(path, top=6):
    import tarfile
    with tarfile.open(path) as tf:
        mem = sorted(tf.getmembers(), key=lambda m: m.size, reverse=True)
    tot = sum(m.size for m in mem)
    return f"{len(mem)} files, {tot/1e6:.1f} MB; largest: " + "; ".join(f"{m.name[-60:]}={m.size/1e6:.1f}MB" for m in mem[:top])


def ids_in(edge):
    out, off = [], None
    from qdrant_edge import ScrollRequest
    while True:
        recs, off = edge.scroll(ScrollRequest(limit=500, offset=off, with_payload=True, with_vector=False))
        out += recs
        if off is None:
            return out


def main():
    for _ in range(30):
        try:
            requests.get(URL + "/readyz", timeout=2)
            break
        except Exception:  # noqa: BLE001
            time.sleep(1)
    requests.delete(URL + f"/collections/{C}")
    req("PUT", f"/collections/{C}", json={
        "vectors": {s.DENSE: {"size": s.DIM, "distance": "Cosine"}},
        "sparse_vectors": {s.SPARSE: {"modifier": "idf"}},
        "sharding_method": "custom",
        "shard_number": 1,
    })
    for z in ("zone_a", "zone_b"):
        req("PUT", f"/collections/{C}/shards", json={"shard_key": z})
    upsert("zone_a", range(100))
    upsert("zone_b", range(100))
    sa = shard_id_for("zone_a")
    print(f"zone_a -> server shard_id {sa}")

    tmp = tempfile.mkdtemp(prefix="edge_sync_")
    try:
        snap = os.path.join(tmp, "full.snapshot")
        full_bytes = download_full(sa, snap)
        print("   full contents:", breakdown(snap))
        edge_dir = os.path.join(tmp, "mirror")
        EdgeShard.unpack_snapshot(snap, edge_dir)
        edge = EdgeShard.load(edge_dir)
        recs = ids_in(edge)
        zones = {r.payload["zone"] for r in recs}
        print(f"[init] full snapshot {full_bytes/1e6:.2f} MB -> edge holds {len(recs)} pts, zones={zones}  "
              f"{'PASS' if len(recs) == 100 and zones == {'zone_a'} else 'FAIL'} per-zone mirror")

        # server-side changes in zone_a: +10 inserts, 5 payload updates, 5 deletes
        upsert("zone_a", range(100, 110))
        req("POST", f"/collections/{C}/points/payload?wait=true",
            json={"payload": {"status": "resolved"}, "points": [uid("zone_a", i) for i in range(5)], "shard_key": "zone_a"})
        req("POST", f"/collections/{C}/points/delete?wait=true",
            json={"points": [uid("zone_a", i) for i in range(10, 15)], "shard_key": "zone_a"})

        # an edge-local write into the mirror shard before applying partial
        local_id = str(uuid.uuid4())
        edge.update(UpdateOperation.upsert_points([Point(local_id, {s.DENSE: s.rvec(999)}, {"zone": "local"})]))
        edge.flush()

        manifest = edge.snapshot_manifest()
        part = os.path.join(tmp, "partial.snapshot")
        part_bytes, ctype = download_partial(sa, manifest, part)
        t0 = time.perf_counter()
        os.makedirs(os.path.join(tmp, "t"), exist_ok=True)
        edge.update_from_snapshot(part, tmp_dir=os.path.join(tmp, "t"))
        apply_ms = (time.perf_counter() - t0) * 1000
        recs = {str(r.id): r for r in ids_in(edge)}
        new_ok = all(uid("zone_a", i) in recs for i in range(100, 110))
        upd_ok = all(recs.get(uid("zone_a", i)) and recs[uid("zone_a", i)].payload.get("status") == "resolved" for i in range(5))
        del_ok = not any(uid("zone_a", i) in recs for i in range(10, 15))
        local_kept = local_id in recs
        print("   partial contents:", breakdown(part))
        print(f"[partial] {part_bytes/1e6:.3f} MB ({ctype}), apply {apply_ms:.0f} ms, edge now {len(recs)} pts")
        print(f"   inserts {'PASS' if new_ok else 'FAIL'} | payload updates {'PASS' if upd_ok else 'FAIL'} | "
              f"deletes propagate {'PASS' if del_ok else 'FAIL'} | edge-local write in mirror survives: {local_kept}")

        # no-change partial: how big is an empty delta?
        manifest = edge.snapshot_manifest()
        nb, _ = download_partial(sa, manifest, os.path.join(tmp, "p2.snapshot"))
        print(f"[partial, no server change] {nb/1e3:.1f} KB")

        # delta after one tiny change on a larger, optimized shard
        upsert("zone_a", range(1000, 6000))
        time.sleep(3)
        manifest = edge.snapshot_manifest()
        nb, _ = download_partial(sa, manifest, os.path.join(tmp, "p3.snapshot"))
        os.makedirs(os.path.join(tmp, "t3"), exist_ok=True)
        edge.update_from_snapshot(os.path.join(tmp, "p3.snapshot"), tmp_dir=os.path.join(tmp, "t3"))
        req("POST", f"/collections/{C}/points/payload?wait=true",
            json={"payload": {"touched": True}, "points": [uid("zone_a", 1001)], "shard_key": "zone_a"})
        time.sleep(2)
        manifest = edge.snapshot_manifest()
        nb1, _ = download_partial(sa, manifest, os.path.join(tmp, "p4.snapshot"))
        full2 = download_full(sa, os.path.join(tmp, "full2.snapshot"))
        print(f"[delta size] +5000 pts partial={nb/1e6:.2f} MB; then 1-payload change partial={nb1/1e6:.3f} MB vs full={full2/1e6:.2f} MB")
        edge.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
