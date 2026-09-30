"""Spike 01: validate Qdrant Edge 0.8.0 engine assumptions at runtime (no server needed).

Each test runs in its own subprocess so a native crash (segfault) is reported as FAIL
instead of killing the harness.

    python spikes/spike_01_engine.py            # run all
    python spikes/spike_01_engine.py <test>     # run one in-process
"""

import datetime as dt
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from qdrant_edge import (
    Bm25,
    Bm25Config,
    DecayKind,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeSparseVectorParams,
    EdgeVectorParams,
    Expression,
    FeedbackItem,
    FeedbackNaiveQuery,
    FieldCondition,
    Filter,
    Formula,
    Fusion,
    Modifier,
    NaiveFeedbackStrategy,
    PayloadSchemaType,
    Point,
    Prefetch,
    Query,
    QueryRequest,
    RangeFloat,
    UpdateMode,
    UpdateOperation,
)

DIM = 8
DENSE = "dense"
SPARSE = "bm25"
BM25 = Bm25(Bm25Config(language="english"))

DOCS = [
    "inverter fault E42 fixed by replacing the DC fuse",
    "tower battery backup low voltage alarm after storm",
    "inverter overheating in afternoon, fan blocked by dust",
    "replaced fibre patch cord, link restored",
    "customer gate code 4471, dog on premises",
    "E42 appears again on site 17 after firmware update",
]


def rvec(seed):
    rnd = random.Random(seed)
    return [rnd.uniform(-1, 1) for _ in range(DIM)]


def config():
    return EdgeConfig(
        vectors={DENSE: EdgeVectorParams(size=DIM, distance=Distance.Cosine)},
        sparse_vectors={SPARSE: EdgeSparseVectorParams(modifier=Modifier.Idf)},
    )


def new_shard(tmp, name="shard"):
    path = os.path.join(tmp, name)
    os.makedirs(path, exist_ok=True)
    return EdgeShard.create(path, config()), path


def seed_points(shard, docs=DOCS, ver=1, days_ago=None):
    now = dt.datetime.now(dt.timezone.utc)
    pts = []
    ids = []
    for i, text in enumerate(docs):
        pid = str(uuid.UUID(int=i + 1))
        ids.append(pid)
        age = (days_ago[i] if days_ago else i)
        pts.append(
            Point(
                pid,
                {DENSE: rvec(i), SPARSE: BM25.embed_document(text)},
                {
                    "text": text,
                    "ver": ver,
                    "observed_at": (now - dt.timedelta(days=age)).isoformat(),
                },
            )
        )
    shard.update(UpdateOperation.upsert_points(pts))
    return ids


# ---------------------------------------------------------------- tests


def t_basic_uuid_roundtrip(tmp):
    shard, path = new_shard(tmp)
    ids = seed_points(shard)
    res = shard.query(QueryRequest(limit=3, query=Query.Nearest(rvec(0), using=DENSE), with_payload=True, with_vector=False))
    assert res and str(res[0].id) == ids[0], f"top hit {res[0].id if res else None}"
    shard.close()
    shard = EdgeShard.load(path)
    got = shard.retrieve([ids[2]], with_payload=True, with_vector=False)
    assert got and got[0].payload["text"] == DOCS[2]
    shard.close()
    return f"uuid ids ok, reload ok, ScoredPoint.version={res[0].version}"


def t_bm25_query(tmp):
    shard, _ = new_shard(tmp)
    seed_points(shard)
    shard.optimize()
    res = shard.query(QueryRequest(limit=3, query=Query.Nearest(BM25.embed_query("E42 inverter"), using=SPARSE), with_payload=True, with_vector=False))
    texts = [r.payload["text"] for r in res]
    assert any("E42" in t for t in texts[:2]), texts
    return f"top={texts[0]!r}"


def t_hybrid_prefetch_weighted_rrf(tmp):
    shard, _ = new_shard(tmp)
    seed_points(shard)
    shard.optimize()
    req = QueryRequest(
        limit=3,
        prefetches=[
            Prefetch(limit=10, query=Query.Nearest(rvec(3), using=DENSE)),
            Prefetch(limit=10, query=Query.Nearest(BM25.embed_query("E42"), using=SPARSE)),
        ],
        query=Fusion.Rrf(k=2, weights=[1.0, 3.0]),
        with_payload=True,
    )
    res = shard.query(req)
    assert len(res) == 3
    return "ranks=" + ", ".join(f"{r.payload['text'][:24]!r}:{r.score:.3f}" for r in res)


def t_formula_recency_decay(tmp):
    shard, _ = new_shard(tmp)
    # identical dense vectors so only recency can separate them
    now = dt.datetime.now(dt.timezone.utc)
    pts = [
        Point(str(uuid.uuid4()), {DENSE: rvec(0)}, {"text": "old", "observed_at": (now - dt.timedelta(days=30)).isoformat()}),
        Point(str(uuid.uuid4()), {DENSE: rvec(0)}, {"text": "new", "observed_at": (now - dt.timedelta(hours=1)).isoformat()}),
    ]
    shard.update(UpdateOperation.upsert_points(pts))
    shard.update(UpdateOperation.create_field_index("observed_at", PayloadSchemaType.Datetime))
    decay = Expression.Decay(
        DecayKind.Exp,
        x=Expression.DatetimeKey("observed_at"),
        target=Expression.Datetime(now.isoformat()),
        scale=7 * 86400.0,
        midpoint=0.5,
    )
    req = QueryRequest(
        limit=2,
        prefetches=[Prefetch(limit=10, query=Query.Nearest(rvec(0), using=DENSE))],
        query=Formula(Expression.Sum([Expression.Variable("$score"), Expression.Mult([Expression.Constant(0.5), decay])])),
        with_payload=True,
    )
    res = shard.query(req)
    order = [r.payload["text"] for r in res]
    assert order == ["new", "old"], order
    return f"new={res[0].score:.3f} old={res[1].score:.3f}"


def t_conditional_upsert_version_guard(tmp):
    shard, _ = new_shard(tmp)
    pid = str(uuid.uuid4())
    shard.update(UpdateOperation.create_field_index("ver", PayloadSchemaType.Integer))

    def put(ver, text):
        guard = Filter(must=[FieldCondition(key="ver", range=RangeFloat(lt=ver))])
        shard.update(UpdateOperation.upsert_points([Point(pid, {DENSE: rvec(1)}, {"ver": ver, "text": text})], condition=guard))
        return shard.retrieve([pid], with_payload=True, with_vector=False)[0].payload

    p = put(5, "v5")
    assert p["ver"] == 5, p  # new point inserted regardless of condition
    p = put(3, "v3-stale")
    assert p["ver"] == 5, f"stale write applied: {p}"
    p = put(7, "v7")
    assert p["ver"] == 7, p
    # InsertOnly must not overwrite
    shard.update(UpdateOperation.upsert_points([Point(pid, {DENSE: rvec(1)}, {"ver": 99})], update_mode=UpdateMode.InsertOnly))
    p = shard.retrieve([pid], with_payload=True, with_vector=False)[0].payload
    assert p["ver"] == 7, p
    return "stale write rejected, newer applied, InsertOnly no-op on existing"


def t_feedback_naive(tmp):
    shard, _ = new_shard(tmp)
    seed_points(shard)
    q = FeedbackNaiveQuery(
        target=rvec(2),
        feedback=[FeedbackItem(rvec(0), 0.9), FeedbackItem(rvec(4), 0.1)],
        strategy=NaiveFeedbackStrategy(a=0.24, b=1.35, c=0.59),
    )
    res = shard.query(QueryRequest(limit=3, query=Query.FeedbackNaive(q, using=DENSE), with_payload=True, with_vector=False))
    assert res
    return "runs; top=" + repr(res[0].payload["text"][:30])


def t_delete_by_id_and_filter(tmp):
    shard, _ = new_shard(tmp)
    ids = seed_points(shard)
    shard.update(UpdateOperation.delete_points([ids[0]]))
    shard.update(UpdateOperation.delete_points_by_filter(Filter(must=[FieldCondition(key="ver", range=RangeFloat(gte=1))])))
    res = shard.query(QueryRequest(limit=10, query=Query.Nearest(rvec(0), using=DENSE)))
    assert res == [], res
    return "all gone"


def t_cross_shard_bm25_incomparable(tmp):
    a, _ = new_shard(tmp, "a")
    b, _ = new_shard(tmp, "b")
    target = "inverter fault E42 fixed by replacing the DC fuse"
    seed_points(a, [target] + [f"E42 note {i}" for i in range(20)])  # E42 common in a
    seed_points(b, [target] + [f"unrelated text {i}" for i in range(20)])  # E42 rare in b
    a.optimize()
    b.optimize()
    q = QueryRequest(limit=1, query=Query.Nearest(BM25.embed_query("E42 inverter"), using=SPARSE), with_payload=True, with_vector=False)
    sa = a.query(q)[0]
    sb = b.query(q)[0]
    assert sa.payload["text"] == sb.payload["text"] == target
    diff = abs(sa.score - sb.score)
    assert diff > 0.05, (sa.score, sb.score)
    return f"same doc, score in A={sa.score:.3f} B={sb.score:.3f} -> raw-score merge is invalid"


def t_thread_offmain_sequential(tmp):
    shard, _ = new_shard(tmp)
    seed_points(shard)
    err = []

    def work():
        try:
            for i in range(200):
                shard.update(UpdateOperation.upsert_points([Point(str(uuid.uuid4()), {DENSE: rvec(i)}, {"i": i})]))
                shard.query(QueryRequest(limit=3, query=Query.Nearest(rvec(i), using=DENSE)))
        except Exception as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert not err, err
    return "200 upsert+query on a worker thread OK"


def t_thread_concurrent(tmp):
    shard, _ = new_shard(tmp)
    seed_points(shard)
    err = []

    def work(k):
        try:
            for i in range(200):
                if k % 2:
                    shard.update(UpdateOperation.upsert_points([Point(str(uuid.uuid4()), {DENSE: rvec(i)}, {"i": i})]))
                else:
                    shard.query(QueryRequest(limit=3, query=Query.Nearest(rvec(i), using=DENSE)))
        except Exception as e:  # noqa: BLE001
            err.append(e)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not err, err
    return "4 threads x200 mixed ops OK"


def t_crash_durability(tmp):
    """Child upserts then hard-exits without flush/close; parent reloads."""
    path = os.path.join(tmp, "crash")
    code = (
        "import os,sys,uuid;sys.path.insert(0,%r);import spike_01_engine as s;"
        "from qdrant_edge import EdgeShard,Point,UpdateOperation;"
        "os.makedirs(%r,exist_ok=True);sh=EdgeShard.create(%r,s.config());"
        "sh.update(UpdateOperation.upsert_points([Point(str(uuid.UUID(int=77)),{s.DENSE:s.rvec(7)},{'k':'v'})]));"
        "os._exit(0)"
    ) % (os.path.dirname(os.path.abspath(__file__)), path, path)
    subprocess.run([sys.executable, "-c", code], check=True)
    shard = EdgeShard.load(path)
    got = shard.retrieve([str(uuid.UUID(int=77))], with_payload=True, with_vector=False)
    assert got and got[0].payload["k"] == "v", got
    return "point survived hard exit without flush (WAL replay)"


def t_second_open_same_dir(tmp):
    shard, path = new_shard(tmp)
    seed_points(shard)
    try:
        EdgeShard.load(path)
    except Exception as e:  # noqa: BLE001
        return f"second open rejected: {type(e).__name__}: {str(e)[:80]}"
    raise AssertionError("second open of same dir succeeded (no lock)")


TESTS = {name: fn for name, fn in globals().items() if name.startswith("t_")}


def run_one(name):
    tmp = tempfile.mkdtemp(prefix="edge_spike_")
    try:
        t0 = time.perf_counter()
        msg = TESTS[name](tmp)
        print(f"PASS {name} ({(time.perf_counter() - t0) * 1000:.0f} ms): {msg}")
    except AssertionError as e:
        print(f"FAIL {name}: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {name}: {type(e).__name__}: {e}")
        sys.exit(2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
        return
    here = os.path.abspath(__file__)
    for name in TESTS:
        p = subprocess.run([sys.executable, here, name], capture_output=True, text=True, timeout=180)
        out = (p.stdout + p.stderr).strip().splitlines()
        if p.returncode not in (0, 1, 2):
            print(f"CRASH {name}: exit code {p.returncode} (native crash?) {out[-1] if out else ''}")
        else:
            print(out[-1] if out else f"?? {name} no output")


if __name__ == "__main__":
    main()
