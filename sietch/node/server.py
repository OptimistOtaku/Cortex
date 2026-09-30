"""A SIETCH robot as a web app: capture, offline memory and recall, the LLM brain, and the comms loop.

create_app() builds one robot. Several robots can share one process and one perception model (sietch/sim/host.py
runs ground, a rover and a scout drone together on one laptop); sietch/node/app.py is the one-robot entry point.
"""

import glob
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sietch.common.geo import parse_pos, where
from sietch.node.brain import Brain
from sietch.node.comms import Comms
from sietch.node.memory import NodeMemory
from sietch.node.store import Store

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_FIELDS = ("ref", "kind", "pos", "state", "tier", "importance", "examined", "superseded", "evidence", "label",
              "camera", "source", "action", "site", "held_reason")


def _public(row):
    row = dict(row)
    row.pop("vec", None)
    return row


def create_app(cfg, perception):
    store = Store(cfg.data_dir, perception.dim)
    mem = NodeMemory(cfg, perception, store)
    brain = Brain(cfg, mem, store)
    comms = Comms(cfg, mem, store, brain)
    if store.replayed:
        store.event("recovery", f"replayed {store.replayed} unflushed writes from the log after an unclean shutdown")
    comms.start()

    app = FastAPI(title=f"SIETCH robot {cfg.node_id}")
    # the Mars simulator (served by ground on another port) drives this robot from the browser
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.node = {"cfg": cfg, "store": store, "mem": mem, "brain": brain, "comms": comms}

    def ingest(data, source, file=None, pos=None, camera=None):
        if pos:
            mem.pose = {**(mem.pose or {}), **pos}
        obs = mem.ingest(data, source=source, file=file, pos=pos, camera=camera)
        obs["brain_woken"] = brain.consider(obs)
        return obs

    @app.post("/api/capture")
    async def capture(request: Request):
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty image")
        h = request.headers
        return _public(ingest(data, h.get("X-Source", "camera"), h.get("X-File"), parse_pos(h.get("X-Pos")),
                              h.get("X-Camera")))

    @app.post("/api/traverse")
    async def traverse(request: Request):
        """Scripted capture for demos/tests: ingest images from a folder as if the robot drove past them."""
        b = await request.json()
        paths = sorted(glob.glob(os.path.join(b["dir"], "**", "*.jp*g"), recursive=True))[: b.get("limit", 1000)]
        woken = 0
        for p in paths:
            folder = os.path.basename(os.path.dirname(p))
            woken += bool(ingest(open(p, "rb").read(), folder, f"{folder}/{os.path.basename(p)}")["brain_woken"])
        return {"ingested": len(paths), "brain_woken": woken}

    @app.post("/api/pose")
    async def pose(request: Request):
        """Where the robot is now (from the simulator), so the brain can say where things are ("38 m NE")."""
        b = await request.json()
        mem.pose = {"x": float(b["x"]), "z": float(b["z"]), "heading": float(b.get("heading", 0.0))}
        return {"ok": True}

    @app.get("/api/memory")
    def memory():
        queue = [_public(r) for r in mem.rank()]
        queued = {r["id"] for r in queue}
        rest = [{"id": str(r.id), **r.payload} for r in store.scroll() if str(r.id) not in queued]
        rest.sort(key=lambda r: r["captured_at"], reverse=True)
        held = [r for r in rest if r.get("state") == "held"]
        return {"intent": mem.intent_text, "queue": queue, "held": held, "sent": [r for r in rest if r not in held]}

    @app.get("/api/map")
    def memory_map():
        """Every memory with where it is and what the gate thinks of it: the simulator draws these as pins."""
        order = {r["id"]: (i + 1, r["value_per_kb"]) for i, r in enumerate(mem.rank())}
        out = []
        for r in store.scroll():
            p, pid = r.payload, str(r.id)
            rank, vpk = order.get(pid, (None, None))
            out.append({"id": pid, **{k: p.get(k) for k in MAP_FIELDS}, "rank": rank, "value_per_kb": vpk,
                        "text": (p.get("text") or p.get("caption") or "")[:160]})
        return {"node": cfg.node_id, "intent": mem.intent_text, "pose": mem.pose, "memories": out}

    @app.get("/api/search")
    def search(q: str, k: int = 8, x: float = None, z: float = None, r: float = None):
        near = ({"x": x, "z": z}, r) if None not in (x, z, r) else None
        res = mem.recall(q, k, near=near)
        for row in res["results"] + res["insights"]:
            row["where"] = where(row.get("pos"), mem.pose)
        return res

    @app.get("/api/near")
    def near(x: float = None, z: float = None, r: float = 40.0):
        """What is remembered around a place (default: here), via a Qdrant Edge geo-radius filter."""
        at = {"x": x, "z": z} if None not in (x, z) else mem.pose
        if not at:
            raise HTTPException(400, "no position yet")
        t0 = time.perf_counter()
        rows = mem.near(at, r)
        return {"ms": round((time.perf_counter() - t0) * 1000, 2), "at": at, "radius_m": r,
                "memories": [{k: v for k, v in row.items() if k not in ("tag_scores", "label_history")} for row in rows]}

    @app.get("/api/brain")
    def brain_state():
        return {**brain.snapshot(), "insights": mem.texts(("insight",), 12), "decisions": mem.texts(("decision",), 12),
                "notes": mem.texts(("note", "answer"), 8)}

    @app.post("/api/ask")
    async def ask(request: Request):
        q = (await request.json())["question"].strip()
        if not q:
            raise HTTPException(400, "empty question")
        if brain.submit("ask", question=q, source="operator"):
            return {"queued": True}
        threading.Thread(target=brain.answer_without_brain, args=(q, None, "operator"), daemon=True).start()
        return {"queued": False, "reason": "reflex-only: answering from recall without the LLM"}

    @app.post("/api/examine/{pid}")
    def examine(pid: str, why: str = "operator asked the brain to look"):
        if not store.get([pid]):
            raise HTTPException(404)
        return {"queued": brain.submit("examine", pid=pid, why=why[:120])}

    @app.post("/api/note")
    async def note(request: Request):
        """A field operator's note: a human insight that syncs like the brain's own."""
        b = await request.json()
        evidence = [e for e in b.get("evidence", []) if store.get([e])]
        n = mem.add_text("note", b["text"], importance=float(b.get("importance", 0.9)), evidence=evidence,
                         source="operator")
        return _public(n)

    @app.get("/api/status")
    def status():
        return {
            "node": cfg.node_id, "model": cfg.model, "intent": mem.intent_text, "memories": store.count(),
            "fleet_known": store.count("fleet"), "bytes_sent": store.bytes_sent(), "comms": comms.status(),
            "brain": {"status": brain.state["status"], "model": brain.state["model"], "queued": brain.q.qsize()},
            "storage": mem.tiers(), "pose": mem.pose, "events": store.events(40), "time": time.time(),
        }

    @app.post("/api/link")
    async def link(request: Request):
        comms.link_up = bool((await request.json())["up"])
        store.event("comms", "link restored" if comms.link_up else "link down: operating offline")
        return {"link_up": comms.link_up}

    @app.post("/api/window_now")
    async def window_now(request: Request):
        """Open a comms window now (the simulator calls this while the relay orbiter is overhead)."""
        body = await request.json() if int(request.headers.get("content-length") or 0) else {}
        comms.open_now(body.get("budget"))
        return {"ok": True}

    @app.get("/img/{pid}/{kind}")
    def img(pid: str, kind: str):
        if kind not in ("orig", "thumb", "full", "best"):
            raise HTTPException(404)
        kinds = ("orig", "full", "thumb") if kind == "best" else (kind,)
        for k in kinds:
            path = os.path.join(store.img_dir, f"{os.path.basename(pid)}_{k}.jpg")
            if os.path.exists(path):
                return FileResponse(path, media_type="image/jpeg")
        raise HTTPException(404)

    app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
    return app

