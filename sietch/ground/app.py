"""Ground control: Qdrant Server holds everything the fleet has sent: pictures, and what the robots' brains concluded.

Robots open comms windows. In each window a robot pulls signed commands (GET /api/uplink) and pushes memories
(POST /api/downlink). Every byte in both directions is counted from the actual request/response bodies.
When two robots' conclusions about the same site contradict (checked by TypeSafe Jev), a dispute opens for mission
control; the resolution supersedes the losing memory on the robot that holds it.

Run: PYTHONUTF8=1 uvicorn sietch.ground.app:app --port 8100
"""

import base64
import json
import os
import threading
import time
import uuid

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient, models

from sietch.common.runtime import load_env, watch_parent

load_env()
watch_parent()

from sietch.common.config import MODELS  # noqa: E402
from sietch.common.db import LockedDB  # noqa: E402
from sietch.common.hlc import HLC  # noqa: E402
from sietch.common.signing import sign  # noqa: E402

QDRANT_URL = os.environ.get("SIETCH_QDRANT", "http://127.0.0.1:6333")
MODEL = os.environ.get("SIETCH_MODEL", "siglip2")
DATA = os.environ.get("SIETCH_GROUND_DATA", os.path.join(os.path.expanduser("~"), ".sietch", "ground"))
JEV_URL = "https://api.typesafe.ai/v1/systemone"
COLLECTION = "fleet"
SAME_SITE = 0.92  # without positions: two sightings from different robots this similar are one site
SAME_SITE_M = 12.0  # with positions: sightings from different robots within this many metres are one site
TEXT_KINDS = ("insight", "decision", "answer", "note")
CONCLUSIONS = ("insight", "note")  # what can contradict another robot's conclusion
HERE = os.path.dirname(os.path.abspath(__file__))
SIM_STATIC = os.path.join(os.path.dirname(HERE), "sim", "static")

os.makedirs(os.path.join(DATA, "img"), exist_ok=True)
DIM = MODELS[MODEL][2]
hlc = HLC("ground")
_lock = threading.RLock()
db = LockedDB(os.path.join(DATA, "ground.db"))
db.executescript(
    """
    CREATE TABLE IF NOT EXISTS commands (id INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT, kind TEXT, body TEXT, hlc TEXT,
        ts REAL, sig TEXT);
    CREATE TABLE IF NOT EXISTS traffic (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, node TEXT, direction TEXT,
        window INTEGER, bytes INTEGER, items INTEGER);
    CREATE TABLE IF NOT EXISTS nodes (node TEXT PRIMARY KEY, last_seen REAL, status TEXT);
    CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, text TEXT);
    CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS disputes (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, site TEXT, a TEXT, b TEXT,
        p REAL, status TEXT, winner TEXT, resolved_ts REAL);
    CREATE TABLE IF NOT EXISTS questions (qid TEXT PRIMARY KEY, ts REAL, node TEXT, question TEXT, answer TEXT,
        cited TEXT, answered_ts REAL, point TEXT);
    """
)
qc = QdrantClient(url=QDRANT_URL, timeout=30)
if not qc.collection_exists(COLLECTION):
    qc.create_collection(COLLECTION, vectors_config={"img": models.VectorParams(size=DIM, distance=models.Distance.COSINE)})
    for f, t in (("node", "keyword"), ("site", "keyword"), ("item", "keyword"), ("kind", "keyword"),
                 ("superseded", "integer"), ("received_at", "float"), ("geo", "geo")):
        qc.create_payload_index(COLLECTION, f, t)

_text_model = None


def text_vec(s):
    global _text_model
    from sietch.node.perception import Perception

    shared = Perception.existing(MODEL)  # one process hosting ground + robots shares one model (run_sim.py)
    if shared:
        return shared.text(s)
    if _text_model is None:
        from fastembed import TextEmbedding

        _text_model = TextEmbedding(model_name=MODELS[MODEL][1])
    v = np.asarray(list(_text_model.embed([s]))[0], dtype=np.float32)
    return v / np.linalg.norm(v)


def dequantize(b64):
    v = np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float32) / 127
    return v / np.linalg.norm(v)


def meta(k, default=None):
    row = db.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
    return row[0] if row else default


def set_meta(k, v):
    db.execute("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (k, v))


def event(kind, text):
    db.execute("INSERT INTO events (ts, kind, text) VALUES (?, ?, ?)", (time.time(), kind, text))


def command(node, kind, body, forge=False):
    ts = hlc.now()
    cid = db.insert("INSERT INTO commands (node, kind, body, hlc, ts) VALUES (?, ?, ?, ?, ?)",
                    (node, kind, json.dumps(body), ts, time.time()))
    sig = "0" * 64 if forge else sign({"id": cid, "node": node, "kind": kind, "body": body, "hlc": ts})
    db.execute("UPDATE commands SET sig = ? WHERE id = ?", (sig, cid))
    return ts


def point_id(item):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, item))


def F(**kv):
    return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in kv.items()])


app = FastAPI(title="SIETCH ground control")


# ---------------------------------------------------------------- comms with robots


@app.post("/api/downlink")
async def downlink(request: Request):
    raw = await request.body()
    msg = json.loads(raw)
    node, window = msg["node"], msg["window"]
    new_conclusions, new_sites = [], []
    with _lock:
        points = []
        for it in msg["items"]:
            hlc.observe(it["hlc"])
            if it["rep"] == "chunk":
                _chunk(it)
                continue
            vec = dequantize(it["emb"])
            pid = point_id(it["id"])
            existing = qc.retrieve(COLLECTION, [pid], with_payload=True)
            old = existing[0].payload if existing else {}
            kind = it.get("kind", "observation")
            payload = {
                "item": it["id"], "ref": it.get("ref"), "node": node, "window": window, "kind": kind,
                "tags": it["tags"], "label": old.get("label", it["label"]), "captured_at": it["captured_at"], "hlc": it["hlc"], "received_at": time.time(),
                "importance": it.get("importance", 0.0), "superseded": old.get("superseded", 0),
                "label_history": old.get("label_history", []),
            }
            if it.get("geo"):
                payload.update(pos=it["pos"], geo=it["geo"])
            if kind in TEXT_KINDS:
                evidence = it.get("evidence", [])
                payload.update({k: it[k] for k in ("text", "action", "reason", "qid", "question", "answer", "source") if k in it})
                payload["evidence"] = evidence
                # a conclusion lives at the site of its first evidence picture (which may not have arrived yet)
                payload["site"] = old.get("site") or (_evidence_site(evidence[0], it.get("evidence_alias", {}), it.get("geo"))
                                                      if evidence else it["id"])
                payload["thumb"] = None
                if kind in CONCLUSIONS and not existing:
                    new_conclusions.append(pid)
                if kind == "answer" and it.get("qid"):
                    db.execute("UPDATE questions SET answer = ?, cited = ?, answered_ts = ?, point = ? WHERE qid = ?",
                               (it.get("answer", it.get("text")), json.dumps(evidence), time.time(), pid, it["qid"]))
                    event("answer", f"{node} answered {it.get('question', '')[:60]!r}: {it.get('answer', '')[:100]!r}")
            else:
                fname = f"{it['id']}_{it['rep']}.jpg"
                with open(os.path.join(DATA, "img", fname), "wb") as f:
                    f.write(base64.b64decode(it["image"]))
                payload.update({"caption": it.get("caption"), "thumb": fname, "full": old.get("full"),
                                "full_progress": old.get("full_progress"),
                                "site": old.get("site") or _site_for(vec, node, it["id"], it.get("geo"))})
                new_sites.append((it["id"], payload["site"]))
            points.append(models.PointStruct(id=pid, vector={"img": vec.tolist()}, payload=payload))
        if points:
            qc.upsert(COLLECTION, points)
        db.execute("INSERT INTO traffic (ts, node, direction, window, bytes, items) VALUES (?, ?, 'down', ?, ?, ?)",
                   (time.time(), node, window, len(raw), len(points)))
        kinds = {}
        for p in points:
            kinds[p.payload["kind"]] = kinds.get(p.payload["kind"], 0) + 1
        event("downlink", f"{node} window {window}: {len(points)} items {kinds or ''}, {len(raw)} B")
    # conclusions that were waiting for their evidence picture get their final site now; then look for disputes
    for item, site in new_sites:
        for c in qc.scroll(COLLECTION, limit=100, scroll_filter=F(site=item))[0]:
            if c.payload["kind"] in CONCLUSIONS and site != item:
                qc.set_payload(COLLECTION, {"site": site}, points=[c.id])
                new_conclusions.append(str(c.id))
    if new_conclusions:
        threading.Thread(target=_check_disputes, args=(new_conclusions,), daemon=True).start()
    return {"ok": True, "received": len(points), "bytes": len(raw)}


def _site_of_item(item):
    rec = qc.retrieve(COLLECTION, [point_id(item)], with_payload=["site"])
    return rec[0].payload["site"] if rec else item


def _evidence_site(item, alias, geo=None):
    """Site of a conclusion's evidence picture. If the picture never came down because another robot had already
    delivered the same scene, the robot names that fleet item (alias) and the conclusion joins its site. If the
    picture simply hasn't come down yet (conclusions travel first), the conclusion joins whatever the fleet already
    knows at that place; otherwise it keeps a provisional site that is fixed when a picture arrives."""
    site = _site_of_item(item)
    if site == item and alias.get(item):
        site = _site_of_item(alias[item])
        if site == alias[item] and not qc.retrieve(COLLECTION, [point_id(alias[item])]):
            site = item
    if site == item and geo:
        site = _site_near(geo) or item
    return site


def _site_near(geo):
    near = qc.scroll(COLLECTION, limit=20, with_payload=["site", "kind"], scroll_filter=models.Filter(
        must=[models.FieldCondition(key="geo", geo_radius=models.GeoRadius(
            center=models.GeoPoint(lon=geo["lon"], lat=geo["lat"]), radius=SAME_SITE_M))]))[0]
    pick = [p for p in near if p.payload.get("kind") == "observation"] or near
    return pick[0].payload["site"] if pick else None


def _chunk(it):
    """Reassemble a full-resolution image streamed across windows."""
    data = base64.b64decode(it["image"])
    part = os.path.join(DATA, "img", f"{it['id']}_full.part")
    with open(part, "r+b" if os.path.exists(part) else "wb") as f:
        f.seek(it["offset"])
        f.write(data)
    got = it["offset"] + len(data)
    pid = point_id(it["id"])
    if got >= it["total"]:
        os.replace(part, os.path.join(DATA, "img", f"{it['id']}_full.jpg"))
        qc.set_payload(COLLECTION, {"full": f"{it['id']}_full.jpg", "full_progress": 100}, points=[pid])
        event("full", f"{it['node']}:{it['id'][:8]} full resolution complete ({it['total']} B)")
    else:
        qc.set_payload(COLLECTION, {"full_progress": round(100 * got / it["total"])}, points=[pid])
        event("full", f"{it['node']}:{it['id'][:8]} full-res chunk {got}/{it['total']} B")


def _site_for(vec, node, item, geo=None):
    """Sightings of the same place from different robots become one site: within SAME_SITE_M when positions are
    known (a scout's aerial view and a rover's close-up look nothing alike), else near-identical pictures."""
    must = [models.FieldCondition(key="kind", match=models.MatchValue(value="observation"))]
    if geo:
        must.append(models.FieldCondition(key="geo", geo_radius=models.GeoRadius(
            center=models.GeoPoint(lon=geo["lon"], lat=geo["lat"]), radius=SAME_SITE_M)))
    hits = qc.query_points(COLLECTION, query=vec.tolist(), using="img", limit=3, with_payload=True,
                           query_filter=models.Filter(
                               must=must,
                               must_not=[models.FieldCondition(key="node", match=models.MatchValue(value=node))])).points
    if hits and geo:
        event("merge", f"{node}:{item[:8]} is the same place as {hits[0].payload['node']}:{hits[0].payload['item'][:8]} "
                       f"(within {SAME_SITE_M:.0f} m)")
        return hits[0].payload["site"]
    if hits and hits[0].score >= SAME_SITE:
        event("merge", f"{node}:{item[:8]} is the same site as {hits[0].payload['node']}:{hits[0].payload['item'][:8]} "
                       f"(sim {hits[0].score:.2f})")
        return hits[0].payload["site"]
    return item


# ---------------------------------------------------------------- disputes


def jev_contradiction(a, b):
    """Probability that two robots reach opposite assessments of one place (Jev noul; wording checked 5/5, findings 2026-09-30)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        return None
    q = {"type": "noul",
         # "opposite assessments" rather than "both cannot be true": robots hedge ("possible evidence of water"), and a
         # hedge technically coexists with "no sign of water" (old wording p=0.19, this one 0.76; findings 2026-09-30)
         "instructions": "Do note A and note B reach opposite assessments of the same place: one sees evidence (even "
                         "tentative) for something that the other says is absent?",
         "criteria": {"true": "Opposite assessments: one finds signs of it, the other says there are none",
                      "false": "Same assessment, complementary details, or about different things"}}
    r = requests.post(JEV_URL, headers={"Authorization": f"Bearer {key}"}, timeout=10,
                      json={"state": {"A": a, "B": b}, "model": "jev-latest", "questions": {"c": q}})
    r.raise_for_status()
    return float(r.json()["answers"]["c"]["noul"])


def _check_disputes(point_ids):
    for pid in dict.fromkeys(point_ids):
        rec = qc.retrieve(COLLECTION, [pid], with_payload=True)
        if not rec or rec[0].payload.get("superseded"):
            continue
        me = rec[0].payload
        others = qc.scroll(COLLECTION, limit=50, with_payload=True, scroll_filter=models.Filter(
            must=[models.FieldCondition(key="site", match=models.MatchValue(value=me["site"])),
                  models.FieldCondition(key="kind", match=models.MatchAny(any=list(CONCLUSIONS)))],
            must_not=[models.FieldCondition(key="node", match=models.MatchValue(value=me["node"])),
                      models.FieldCondition(key="superseded", match=models.MatchValue(value=1))]))[0]
        for o in others:
            pair = sorted([pid, str(o.id)])
            if db.execute("SELECT 1 FROM disputes WHERE a = ? AND b = ?", tuple(pair)).fetchone():
                continue
            try:
                p = jev_contradiction(o.payload["text"], me["text"])
            except requests.RequestException as e:
                event("dispute", f"contradiction check unavailable ({type(e).__name__}); compare manually: "
                                 f"{o.payload['node']} vs {me['node']} at site {me['site'][:8]}")
                continue
            if p is None:
                event("dispute", "no TYPESAFE_API_KEY: contradiction checks disabled")
                return
            if p >= 0.5:
                status = "open" if p >= 0.65 else "unsure"
                db.execute("INSERT INTO disputes (ts, site, a, b, p, status) VALUES (?, ?, ?, ?, ?, ?)",
                           (time.time(), me["site"], pair[0], pair[1], p, status))
                event("dispute", f"{o.payload['node']} and {me['node']} disagree about site {me['site'][:8]} "
                                 f"(p={p:.2f}, {status}): {o.payload['text'][:60]!r} vs {me['text'][:60]!r}")
            else:
                event("dispute", f"{o.payload['node']} and {me['node']} agree/compatible at site {me['site'][:8]} (p={p:.2f})")


@app.get("/api/disputes")
def disputes():
    rows = db.execute("SELECT id, ts, site, a, b, p, status, winner FROM disputes ORDER BY id DESC").fetchall()
    out = []
    for did, ts, site, a, b, p, status, winner in rows:
        recs = {str(r.id): r.payload for r in qc.retrieve(COLLECTION, [a, b], with_payload=True)}
        evidence = qc.scroll(COLLECTION, limit=6, with_payload=True,
                             scroll_filter=models.Filter(must=[models.FieldCondition(key="site", match=models.MatchValue(value=site)),
                                                               models.FieldCondition(key="kind", match=models.MatchValue(value="observation"))]))[0]
        out.append({"id": did, "ts": ts, "site": site, "p": p, "status": status, "winner": winner,
                    "a": {"point": a, **recs.get(a, {})}, "b": {"point": b, **recs.get(b, {})},
                    "evidence": [{"point": str(e.id), **e.payload} for e in evidence]})
    return out


@app.post("/api/disputes/{did}/resolve")
async def resolve(did: int, request: Request):
    winner = (await request.json())["winner"]  # "a" | "b"
    row = db.execute("SELECT a, b, status FROM disputes WHERE id = ?", (did,)).fetchone()
    if not row:
        raise HTTPException(404)
    a, b, _ = row
    win, lose = (a, b) if winner == "a" else (b, a)
    recs = {str(r.id): r.payload for r in qc.retrieve(COLLECTION, [win, lose], with_payload=True)}
    w, l = recs[win], recs[lose]
    with _lock:
        ts = command(l["node"], "supersede", {"item": l["item"], "by": "mission control", "winner": w["text"]})
        qc.set_payload(COLLECTION, {"superseded": 1, "superseded_by": {"text": w["text"], "hlc": ts}}, points=[lose])
        db.execute("UPDATE disputes SET status = 'resolved', winner = ?, resolved_ts = ? WHERE id = ?", (win, time.time(), did))
        event("dispute", f"resolved #{did}: {w['node']} is right; {l['node']}'s {l.get('ref')} superseded "
                         f"(reaches {l['node']} at its next window)")
    return {"ok": True}


# ---------------------------------------------------------------- uplink


@app.get("/api/uplink")
def uplink(node: str, since: int = 0, fleet_since: float = 0.0, status: str = ""):
    """Signed commands for one robot since its watermark, plus fleet knowledge (what other robots delivered)."""
    with _lock:
        db.execute("INSERT INTO nodes (node, last_seen, status) VALUES (?, ?, ?) "
                   "ON CONFLICT(node) DO UPDATE SET last_seen = excluded.last_seen, status = excluded.status",
                   (node, time.time(), status))
        rows = db.execute("SELECT id, node, kind, body, hlc, sig FROM commands WHERE id > ? AND node IN (?, '*') ORDER BY id",
                          (since, node)).fetchall()
        cmds = [{"id": i, "node": n, "kind": k, "body": json.loads(b), "hlc": h, "sig": s} for i, n, k, b, h, s in rows]
        fleet = qc.scroll(COLLECTION, limit=500, with_vectors=True, with_payload=True, scroll_filter=models.Filter(
            must=[models.FieldCondition(key="received_at", range=models.Range(gt=fleet_since))],
            must_not=[models.FieldCondition(key="node", match=models.MatchValue(value=node))]))[0]
        fleet_items = []
        for p in fleet:
            q = np.clip(np.round(np.asarray(p.vector["img"]) * 127), -127, 127).astype(np.int8)
            fleet_items.append({"id": p.payload["item"], "node": p.payload["node"], "tags": p.payload["tags"],
                                "kind": p.payload.get("kind", "observation"),
                                "emb": base64.b64encode(q.tobytes()).decode(), "received_at": p.payload["received_at"]})
        body = {"commands": cmds, "fleet": fleet_items, "ground_hlc": hlc.now()}
        raw = json.dumps(body).encode()
        db.execute("INSERT INTO traffic (ts, node, direction, window, bytes, items) VALUES (?, ?, 'up', 0, ?, ?)",
                   (time.time(), node, len(raw), len(cmds) + len(fleet_items)))
    return Response(raw, media_type="application/json")


# ---------------------------------------------------------------- mission control actions


@app.post("/api/intent")
async def set_intent(request: Request):
    text = (await request.json())["text"].strip()
    if not text:
        raise HTTPException(400, "empty intent")
    with _lock:
        set_meta("intent", text)
        ts = command("*", "intent", {"text": text})
        event("intent", f"mission intent -> {text!r} (reaches each robot at its next window)")
    return {"ok": True, "hlc": ts}


@app.post("/api/ask")
async def ask(request: Request):
    b = await request.json()
    q, node = b["question"].strip(), b["node"]
    qid = str(uuid.uuid4())
    with _lock:
        command(node, "ask", {"qid": qid, "question": q})
        db.execute("INSERT INTO questions (qid, ts, node, question) VALUES (?, ?, ?, ?)", (qid, time.time(), node, q))
        event("ask", f"asked {node}: {q!r} (goes up at its next window; the robot answers offline)")
    return {"ok": True, "qid": qid}


@app.get("/api/questions")
def questions():
    rows = db.execute("SELECT qid, ts, node, question, answer, cited, answered_ts FROM questions ORDER BY ts DESC LIMIT 30").fetchall()
    return [{"qid": q, "ts": ts, "node": n, "question": qu, "answer": a, "cited": json.loads(c or "[]"),
             "answered_ts": at, "latency_s": round(at - ts, 1) if at else None} for q, ts, n, qu, a, c, at in rows]


@app.post("/api/request_full")
async def request_full(request: Request):
    b = await request.json()
    rec = qc.retrieve(COLLECTION, [b["point"]], with_payload=True)
    if not rec:
        raise HTTPException(404)
    p = rec[0].payload
    if p.get("kind", "observation") != "observation":
        raise HTTPException(400, "only observations have images")
    command(p["node"], "request_full", {"item": p["item"]})
    event("request", f"requested full resolution of {p['node']}:{p['item'][:8]} (arrives in a later window)")
    return {"ok": True}


@app.post("/api/request_evidence")
async def request_evidence(request: Request):
    """For a conclusion that arrived before its pictures: ask the robot to prioritise the cited evidence."""
    b = await request.json()
    rec = qc.retrieve(COLLECTION, [b["point"]], with_payload=True)
    if not rec:
        raise HTTPException(404)
    p = rec[0].payload
    for e in p.get("evidence", []):
        command(p["node"], "request_full", {"item": e})
    event("request", f"requested the evidence behind {p.get('ref')} from {p['node']} ({len(p.get('evidence', []))} items)")
    return {"ok": True}


@app.post("/api/relabel")
async def relabel(request: Request):
    b = await request.json()
    rec = qc.retrieve(COLLECTION, [b["point"]], with_payload=True)
    if not rec:
        raise HTTPException(404)
    p = rec[0].payload
    with _lock:
        ts = command(p["node"], "relabel", {"item": p["item"], "label": b["label"], "by": "ground"})
        hist = p.get("label_history", []) + [{"label": p["label"], "superseded_by": b["label"], "hlc": ts}]
        site_pts = qc.scroll(COLLECTION, limit=100, scroll_filter=models.Filter(
            must=[models.FieldCondition(key="site", match=models.MatchValue(value=p["site"])),
                  models.FieldCondition(key="kind", match=models.MatchValue(value="observation"))]))[0]
        qc.set_payload(COLLECTION, {"label": b["label"], "label_history": hist}, points=[x.id for x in site_pts])
        for x in site_pts:
            if x.payload["node"] != p["node"]:
                command(x.payload["node"], "relabel", {"item": x.payload["item"], "label": b["label"], "by": "ground"})
        event("relabel", f"site {p['site'][:8]}: {p['label']!r} superseded by {b['label']!r} on {len(site_pts)} sightings")
    return {"ok": True, "hlc": ts}


@app.post("/api/discard")
async def discard(request: Request):
    b = await request.json()
    rec = qc.retrieve(COLLECTION, [b["point"]], with_payload=True)
    if not rec:
        raise HTTPException(404)
    p = rec[0].payload
    with _lock:
        command(p["node"], "discard", {"item": p["item"]})
        qc.delete(COLLECTION, points_selector=models.FilterSelector(filter=F(item=p["item"])))
        event("discard", f"{p['node']}:{p['item'][:8]} discarded; tombstone queued for the robot")
    return {"ok": True}


@app.post("/api/redteam/forge")
async def forge(request: Request):
    """Demo only: queue a discard with a bad signature, as an attacker spoofing mission control would."""
    b = await request.json()
    node = b["node"]
    victim = qc.scroll(COLLECTION, limit=1, with_payload=True, scroll_filter=F(node=node, kind="observation"))[0]
    item = victim[0].payload["item"] if victim else "00000000"
    command(node, "discard", {"item": item}, forge=True)
    event("security", f"RED TEAM: forged 'discard {item[:8]}' queued for {node} with a bad signature")
    return {"ok": True, "item": item}


# ---------------------------------------------------------------- views


def _view(p):
    return {"point": str(p.id), "score": getattr(p, "score", None), **p.payload}


@app.get("/api/feed")
def feed(kind: str = "", limit: int = 60):
    must = [models.FieldCondition(key="kind", match=models.MatchAny(any=kind.split(",")))] if kind else []
    pts = qc.scroll(COLLECTION, limit=1000, with_payload=True, scroll_filter=models.Filter(must=must))[0]
    pts.sort(key=lambda p: p.payload["received_at"], reverse=True)
    return [_view(p) for p in pts[:limit]]


@app.get("/api/search")
def search(q: str, limit: int = 12):
    t0 = time.perf_counter()
    v = text_vec(q)
    out = {}
    for group, kinds in (("results", ["observation"]), ("conclusions", list(TEXT_KINDS))):
        out[group] = [_view(p) for p in qc.query_points(
            COLLECTION, query=v.tolist(), using="img", limit=limit if group == "results" else 5, with_payload=True,
            query_filter=models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchAny(any=kinds))])).points]
    return {"ms": round((time.perf_counter() - t0) * 1000, 1), **out}


@app.get("/api/metrics")
def metrics():
    down, items = db.execute("SELECT COALESCE(SUM(bytes), 0), COALESCE(SUM(items), 0) FROM traffic "
                             "WHERE direction = 'down'").fetchone()
    up = db.execute("SELECT COALESCE(SUM(bytes), 0) FROM traffic WHERE direction = 'up'").fetchone()[0]
    open_disputes = db.execute("SELECT COUNT(*) FROM disputes WHERE status != 'resolved'").fetchone()[0]
    conclusions = qc.count(COLLECTION, count_filter=models.Filter(
        must=[models.FieldCondition(key="kind", match=models.MatchAny(any=list(CONCLUSIONS)))])).count
    pictures = qc.count(COLLECTION, count_filter=F(kind="observation")).count
    return {"downlink_bytes": down, "items": items, "uplink_bytes": up, "conclusions": conclusions,
            "pictures": pictures, "open_disputes": open_disputes,
            "intent": meta("intent", "anything unusual or interesting")}


@app.get("/api/nodes")
def nodes():
    rows = db.execute("SELECT node, last_seen, status FROM nodes ORDER BY node").fetchall()
    return [{"node": n, "last_seen": t, "ago_s": round(time.time() - t, 1), "status": json.loads(s or "{}")}
            for n, t, s in rows]


@app.get("/api/events")
def events(limit: int = 60):
    rows = db.execute("SELECT ts, kind, text FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"ts": ts, "kind": k, "text": t} for ts, k, t in rows]


@app.get("/img/{name}")
def img(name: str):
    path = os.path.join(DATA, "img", os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


if os.path.isdir(SIM_STATIC):
    app.mount("/sim", StaticFiles(directory=SIM_STATIC, html=True), name="sim")
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
