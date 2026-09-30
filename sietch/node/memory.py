"""A robot's semantic memory: what it has seen and what its brain has concluded, plus the sync gate.

Memory kinds, all in one Qdrant Edge shard and one SigLIP2 image-text space (so a brain-written sentence sits next
to the images it is about):
  observation - a camera frame: image embedding, zero-shot tags, caption once the brain has looked
  insight     - the brain's conclusion: text, importance, evidence ids, site
  decision    - what the brain chose to do and why, with the memories it cited (its audit trail)
  answer      - the brain's answer to a question from mission control
  note        - a human field operator's note

The sync gate ranks unsent memories by value per kilobyte with Qdrant Formula queries. Text memories are a few hundred
bytes against 6-30 KB for an image, so the robot's conclusions reach ground before its pictures; ground pulls the
evidence on demand.
"""

import base64
import io
import os
import re
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps
from qdrant_edge import (
    Bm25,
    Bm25Config,
    Expression,
    FieldCondition,
    Filter,
    Formula,
    Fusion,
    GeoPoint,
    GeoRadius,
    MatchValue,
    Prefetch,
    Query,
    QueryRequest,
)

from sietch.common.geo import to_geo, where
from sietch.common.hlc import HLC
from sietch.node.perception import jpeg

DEFAULT_INTENT = "anything unusual or interesting"
E = Expression
TEXT_KINDS = ("insight", "decision", "answer", "note")
MIN_COST_KB = 6.0  # a heavily blurred thumbnail compresses to ~2 KB; don't let cheapness alone win
MIN_COST_KB_TEXT = 1.0  # text memories really are this small; that's the point
CHUNK_OVERHEAD = 700  # JSON envelope + fields around one full-res chunk
# Flags are stored as integers 0/1: on qdrant-edge-py 0.8.0 a boolean MatchValue(True) never matches (see findings).
REQUESTED = Filter(must=[FieldCondition(key="requested_full", match=MatchValue(value=1))])
ID_RE = re.compile(r"\b(obs|ins|dec|ans|note)-([0-9a-f]{8})\b")
PREFIX = {"observation": "obs", "insight": "ins", "decision": "dec", "answer": "ans", "note": "note"}


def cost_kb(nbytes, kind="observation"):
    return max(nbytes / 1000, MIN_COST_KB_TEXT if kind in TEXT_KINDS else MIN_COST_KB)


def quantize(vec):
    """768 floats in [-1, 1] -> 768 int8 bytes, base64. 4x smaller on the wire than float32."""
    q = np.clip(np.round(np.asarray(vec) * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode()


def dequantize(b64):
    v = np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float32) / 127
    return v / np.linalg.norm(v)


def ref(kind, pid):
    """Short id the brain and humans use in text, e.g. obs-1a2b3c4d."""
    return f"{PREFIX.get(kind, 'obs')}-{pid[:8]}"


def _kind_is(*kinds):
    return [FieldCondition(key="kind", match=MatchValue(value=k)) for k in kinds]


NOT_SUPERSEDED = FieldCondition(key="superseded", match=MatchValue(value=1))


def _within(pos, radius_m):
    g = to_geo(pos["x"], pos["z"])
    return FieldCondition(key="geo", geo_radius=GeoRadius(GeoPoint(g["lon"], g["lat"]), float(radius_m)))


class NodeMemory:
    def __init__(self, cfg, perception, store):
        self.cfg, self.p, self.store = cfg, perception, store
        self.hlc = HLC(cfg.node_id, last=store.meta("hlc"))
        self.bm25 = Bm25(Bm25Config(language="english"))
        self.intent_text = store.meta("intent_text", DEFAULT_INTENT)
        self.intent_vec = self.p.text(self.intent_text)
        self.pose = None  # {"x", "z", "heading"} in simulator metres, when the robot knows where it is

    def _tick(self):
        ts = self.hlc.now()
        self.store.set_meta("hlc", ts)
        return ts

    def _path(self, pid, kind):
        return os.path.join(self.store.img_dir, f"{pid}_{kind}.jpg")

    def _base(self, kind, pid, **extra):
        return {
            "node": self.cfg.node_id, "hlc": self._tick(), "captured_at": datetime.now(timezone.utc).isoformat(),
            "kind": kind, "ref": ref(kind, pid), "importance": 0.0, "superseded": 0, "site": pid,
            "bytes_full": 0, "full_size": 0, "full_sent": 0,
            "state": "new", "requested_full": 0, "hits": 0, "influence": 0.0,
            "label_history": [], **extra,
        }

    # ------------------------------------------------------------ capture

    def ingest(self, image_bytes, source="camera", file=None, pos=None, camera=None):
        t0 = time.perf_counter()
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        vec = self.p.image(pil)
        tags, tag_scores = self.p.tags(vec)
        thumb = jpeg(pil, self.cfg.thumb_px, self.cfg.thumb_quality)
        full = jpeg(pil, self.cfg.full_px, self.cfg.full_quality)
        pid = str(uuid.uuid4())
        open(self._path(pid, "orig"), "wb").write(jpeg(pil, self.cfg.full_px, 90))  # stays on this device
        open(self._path(pid, "thumb"), "wb").write(thumb)
        if full:
            open(self._path(pid, "full"), "wb").write(full)
        novelty, nearest = self._novelty(vec)
        bytes_thumb = len(thumb) * 4 // 3 + 1024 + 400  # base64 thumb + int8 embedding + metadata, refined on send
        payload = self._base(
            "observation", pid, source=source, file=file, tags=tags, tag_scores=tag_scores, caption=" ".join(tags),
            label=tags[0],
            bytes_thumb=bytes_thumb, bytes_full=(len(full) * 4 // 3 + 400) if full else 0,
            full_size=len(full) if full else 0, bytes_next=bytes_thumb, cost_kb=cost_kb(bytes_thumb),
            novelty=round(novelty, 4), nearest=nearest, tier="full", examined=0, camera=camera,
            **self._located(pos),
        )
        self.store.upsert(pid, vec, payload, sparse=self.bm25.embed_document(payload["caption"]))
        self.store.event("capture", f"{ref('observation', pid)} {tags} novelty={novelty:.2f}")
        self.enforce_budget()
        return {"id": pid, **payload, "ms": round((time.perf_counter() - t0) * 1000)}

    def add_text(self, kind, text, importance=0.5, evidence=(), source="brain", **extra):
        """Store a text memory (insight, decision, answer, note). Returns its payload with id."""
        assert kind in TEXT_KINDS, kind
        text = " ".join(str(text).split())[:600]
        vec = self.p.text(text)
        pid = str(uuid.uuid4())
        evidence = [e for e in dict.fromkeys(evidence) if e]
        site = self._site_of(evidence[0]) if evidence else pid
        first = self.store.get([evidence[0]]) if evidence else []
        extra = {**self._located(first[0].payload.get("pos") if first else None), **extra}
        est = len(text.encode()) + 1024 + 500  # text + int8 embedding + fields
        payload = self._base(kind, pid, source=source, text=text, caption=text, label=kind, tags=[kind],
                             importance=float(max(0.0, min(1.0, importance))), evidence=evidence, site=site,
                             bytes_thumb=est, bytes_next=est, cost_kb=cost_kb(est, kind), novelty=1.0, nearest=None,
                             **extra)
        self.store.upsert(pid, vec, payload, sparse=self.bm25.embed_document(text))
        self.store.event(kind, f"{ref(kind, pid)} ({source}, importance {payload['importance']:.2f}): {text[:120]}")
        return {"id": pid, **payload}

    @staticmethod
    def _located(pos):
        return {"pos": pos, "geo": to_geo(pos["x"], pos["z"])} if pos else {}

    def _site_of(self, pid):
        rec = self.store.get([pid])
        return rec[0].payload.get("site", pid) if rec else pid

    def _novelty(self, vec):
        """1 - max similarity to anything this node or the fleet (via ground) already holds."""
        best, nearest = 0.0, None
        for shard in ("memory", "fleet"):
            flt = Filter(must=_kind_is("observation")) if shard == "memory" else None
            hits = self.store.query(QueryRequest(limit=1, filter=flt, query=Query.Nearest(list(map(float, vec)), using="img")), shard)
            if hits and hits[0].score > best:
                best, nearest = float(hits[0].score), f"{shard}:{str(hits[0].id)[:8]}"
        return max(0.0, 1.0 - best), nearest

    # ------------------------------------------------------------ recall (offline)

    def _hybrid(self, text, k, kinds, dense, sparse, near=None):
        flt = Filter(should=_kind_is(*kinds), must_not=[NOT_SUPERSEDED], must=[_within(*near)] if near else None)
        prefetches = [Prefetch(limit=50, filter=flt, query=Query.Nearest(dense, using="img"))]
        if sparse.indices:
            prefetches.append(Prefetch(limit=50, filter=flt, query=Query.Nearest(sparse, using="tags")))
        return self.store.query(QueryRequest(
            limit=k, prefetches=prefetches, query=Fusion.Rrf(k=2, weights=[3.0, 1.0][:len(prefetches)]),
            with_payload=True, with_vector=False,
        ))

    def recall(self, text, k=6, k_text=4, text_kinds=TEXT_KINDS, near=None):
        """Hybrid (dense + BM25, RRF) over observations and, separately, over what the brain/operators wrote.

        Kinds are queried separately because text-to-text cosine runs much higher than text-to-image in SigLIP2,
        so a single ranking would bury every image under the insights. near=(pos, radius_m) adds a geo filter.
        """
        t0 = time.perf_counter()
        dense = list(map(float, self.p.text(text)))
        sparse = self.bm25.embed_query(text)
        t_embed = time.perf_counter()
        obs = self._hybrid(text, k, ("observation",), dense, sparse, near)
        txt = self._hybrid(text, k_text, text_kinds, dense, sparse, near) if k_text else []
        t_done = time.perf_counter()
        for r in obs[:3]:  # being retrieved makes a memory more influential
            hits = r.payload.get("hits", 0) + 1
            self.store.set_payload([str(r.id)], {"hits": hits, "influence": min(1.0, hits / 5)})
        row = lambda r: {"id": str(r.id), "score": round(r.score, 4), **r.payload}
        return {"ms": round((t_done - t0) * 1000, 1), "embed_ms": round((t_embed - t0) * 1000, 1),
                "qdrant_ms": round((t_done - t_embed) * 1000, 2),
                "results": [row(r) for r in obs], "insights": [row(r) for r in txt]}

    search = recall

    def near(self, pos, radius_m=40.0, limit=30):
        """Everything remembered within radius_m of pos (a Qdrant Edge geo filter), nearest first."""
        recs = self.store.scroll(Filter(must=[_within(pos, radius_m)], must_not=[NOT_SUPERSEDED]))
        rows = [{"id": str(r.id), **r.payload, "where": where(r.payload.get("pos"), pos)} for r in recs]
        rows.sort(key=lambda r: (r["pos"]["x"] - pos["x"]) ** 2 + (r["pos"]["z"] - pos["z"]) ** 2)
        return rows[:limit]

    def resolve_refs(self, text):
        """Full ids for short refs like [obs-1a2b3c4d] that appear in text."""
        wanted = {m.group(2) for m in ID_RE.finditer(text or "")}
        if not wanted:
            return []
        return [str(r.id) for r in self.store.scroll() if str(r.id)[:8] in wanted]

    def strip_unknown_refs(self, text):
        """Remove citations of memories that do not exist (a small LLM sometimes copies ids from its prompt)."""
        known = {i[:8] for i in self.resolve_refs(text)}
        out = ID_RE.sub(lambda m: m.group(0) if m.group(2) in known else "\x00", text or "")
        out = re.sub(r"\s*\[\x00\]|\x00|\s*\[(?:(?:obs|ins|dec|ans|note)-)?<[^>\]]*>\]", "", out)
        return re.sub(r"\s*\[(?:its|this|the|recalled|another)? ?id\]", "", out, flags=re.I).strip()  # "[its id]"

    def texts(self, kinds=TEXT_KINDS, limit=30):
        recs = self.store.scroll(Filter(should=_kind_is(*kinds)))
        rows = [{"id": str(r.id), **r.payload} for r in recs]
        rows.sort(key=lambda r: r["hlc"], reverse=True)
        return rows[:limit]

    # ------------------------------------------------------------ brain/operator updates on memories

    def set_caption(self, pid, caption):
        rec = self.store.get([pid], with_vector=True)
        if not rec:
            return
        p = dict(rec[0].payload)
        p["caption"] = " ".join(caption.split())[:400]
        p["examined"] = 1
        vec = rec[0].vector["img"] if isinstance(rec[0].vector, dict) else rec[0].vector
        self.store.upsert(pid, vec, p, sparse=self.bm25.embed_document(" ".join(p["tags"]) + " " + p["caption"]))

    def set_importance(self, pid, importance):
        self.store.set_payload([pid], {"importance": float(max(0.0, min(1.0, importance)))})

    def supersede(self, pid, by, winner_text, remote_hlc):
        rec = self.store.get([pid])
        if not rec:
            return False
        if remote_hlc:
            self.store.set_meta("hlc", self.hlc.observe(remote_hlc))
        self.store.set_payload([pid], {"superseded": 1, "superseded_by": {"by": by, "text": winner_text, "hlc": remote_hlc}})
        self.store.event("supersede", f"{rec[0].payload.get('ref', pid[:8])} superseded by {by}: {winner_text[:100]!r}; "
                                      f"the brain no longer recalls it")
        return True

    # ------------------------------------------------------------ mission intent

    def set_intent(self, text, remote_hlc=None):
        if remote_hlc:
            self.store.set_meta("hlc", self.hlc.observe(remote_hlc))
        self.intent_text = text
        self.intent_vec = self.p.text(text)
        self.store.set_meta("intent_text", text)
        self.store.event("intent", f"mission intent -> {text!r}; queue re-ranked locally")

    # ------------------------------------------------------------ sync gate

    @staticmethod
    def _pending(kinds=None):
        return Filter(must=[FieldCondition(key="state", match=MatchValue(value="new"))],
                      should=_kind_is(*kinds) if kinds else None)

    def refresh_novelty(self):
        """Novelty vs the fleet changes whenever ground tells us what other rovers sent."""
        for rec in self.store.scroll(self._pending(("observation",)), with_vector=True):
            nov, nearest = self._novelty_excluding(self._vec(rec), str(rec.id))
            if abs(nov - rec.payload.get("novelty", 0)) > 1e-3:
                self.store.set_payload([str(rec.id)], {"novelty": round(nov, 4), "nearest": nearest})

    def _novelty_excluding(self, vec, pid):
        best, nearest = 0.0, None
        for shard in ("memory", "fleet"):
            flt = Filter(must=_kind_is("observation")) if shard == "memory" else None
            for h in self.store.query(QueryRequest(limit=2, filter=flt, query=Query.Nearest(list(map(float, vec)), using="img")), shard):
                if str(h.id) != pid and h.score > best:
                    best, nearest = float(h.score), f"{shard}:{str(h.id)[:8]}"
        return max(0.0, 1.0 - best), nearest

    def rank(self, limit=200):
        """Downlink queue order: Qdrant Formula queries rank unsent memories by value per KB, where
        value = w_intent * relevance^2 + w_novelty * novelty + w_influence * influence + w_importance * importance.
        Relevance is the intent similarity min-max normalised within each group (observations, text memories):
        SigLIP cosine scores are small and bunched, and text-text runs higher than text-image.
        Full-resolution requests from ground come first ("demand").
        """
        rows = [{**self._row(r, None), "demand": True} for r in self.store.scroll(REQUESTED, with_vector=True)]
        ranked = []
        for kinds in (("observation",), TEXT_KINDS):
            pending = self.store.scroll(self._pending(kinds), with_vector=True)
            if not pending:
                continue
            sims = [float(np.asarray(self._vec(r)) @ self.intent_vec) for r in pending]
            lo, hi = min(sims), max(sims)
            span = hi - lo if hi - lo > 1e-6 else 1.0
            cfg = self.cfg
            relevance = E.Div(E.Sum([E.Variable("$score"), E.Constant(-lo)]), E.Constant(span), 0.0)
            value = E.Sum([
                E.Mult([E.Constant(cfg.w_intent), E.Pow(relevance, E.Constant(2.0))]),  # sharpen: strong matches dominate
                E.Mult([E.Constant(cfg.w_novelty), E.Variable("novelty")]),
                E.Mult([E.Constant(cfg.w_influence), E.Variable("influence")]),
                E.Mult([E.Constant(cfg.w_importance), E.Variable("importance")]),
            ])
            per_kb = E.Div(value, E.Variable("cost_kb"), 0.0)
            res = self.store.query(QueryRequest(
                limit=limit,
                prefetches=[Prefetch(limit=max(limit, len(pending)), filter=self._pending(kinds),
                                     query=Query.Nearest(list(map(float, self.intent_vec)), using="img"))],
                query=Formula(per_kb), with_payload=True, with_vector=True,
            ))
            ranked += [self._row(r, r.score, lo, span) for r in res]
        ranked.sort(key=lambda r: r["value_per_kb"], reverse=True)
        return (rows + ranked)[:limit]

    @staticmethod
    def _vec(r):
        return r.vector["img"] if isinstance(r.vector, dict) else r.vector

    def _row(self, r, value_per_kb, lo=0.0, span=1.0):
        vec = np.asarray(self._vec(r), dtype=np.float32)
        sim = float(vec @ self.intent_vec)
        return {
            "id": str(r.id), "vec": vec, "value_per_kb": None if value_per_kb is None else round(value_per_kb, 4),
            "intent_sim": round(sim, 4), "relevance": round((sim - lo) / span, 3), **r.payload,
        }

    def plan_window(self, budget):
        """Fill a byte budget. Returns (selected, skipped-with-reason)."""
        self.refresh_novelty()
        selected, skipped, used = [], [], 0
        for it in self.rank():
            if it.get("demand"):  # stream the requested full-res image in chunks across windows
                remaining = it["full_size"] - it.get("full_sent", 0)
                room = (budget - used - CHUNK_OVERHEAD) * 3 // 4  # base64 inflates by 4/3
                n = min(remaining, room)
                if n <= 0:
                    skipped.append((it, "full-res request waits: no room left in this window"))
                    continue
                cost = n * 4 // 3 + CHUNK_OVERHEAD
                selected.append({**it, "rep": "chunk", "offset": it.get("full_sent", 0), "n": n, "cost": cost})
                used += cost
                continue
            text = it.get("kind") in TEXT_KINDS
            rep, cost = ("text" if text else "thumb"), it["bytes_thumb"]
            if used + cost > budget:
                skipped.append((it, "does not fit remaining budget"))
                continue
            if not text and not os.path.exists(self._path(it["id"], "thumb")):
                skipped.append((it, "evidence evicted from storage"))
                continue
            # redundancy checks apply to pictures only: two contradicting insights can be textually close,
            # and holding one back would hide the conflict from mission control
            if not text:
                dup = next((s for s in selected if s["rep"] == "thumb" and float(s["vec"] @ it["vec"]) > self.cfg.near_duplicate), None)
                if dup:
                    skipped.append((it, f"near-duplicate of {dup['id'][:8]} in this window"))
                    continue
                if it.get("novelty", 1) < 1 - self.cfg.near_duplicate and str(it.get("nearest", "")).startswith("fleet:"):
                    who = self._fleet_node(it["nearest"])
                    reason = f"{who} already delivered this to ground"
                    self.store.set_payload([it["id"]], {"state": "held", "held_reason": reason})
                    self.store.event("gate", f"{it.get('ref', it['id'][:8])} held back: {reason} (0 bytes spent)")
                    skipped.append((it, reason))
                    continue
            selected.append({**it, "rep": rep, "cost": cost})
            used += cost
        return selected, skipped

    def _fleet_node(self, nearest):
        short = nearest.split(":", 1)[1]
        for rec in self.store.scroll(shard="fleet"):
            if str(rec.id).startswith(short):
                return rec.payload.get("node", "another rover")
        return "another rover"

    def _fleet_alias(self, pid):
        rec = self.store.get([pid])
        p = rec[0].payload if rec else {}
        nearest = str(p.get("nearest") or "")
        if not nearest.startswith("fleet:") or p.get("novelty", 1) >= 1 - self.cfg.near_duplicate:
            return None
        short = nearest.split(":", 1)[1]
        return next((str(r.id) for r in self.store.scroll(shard="fleet") if str(r.id).startswith(short)), None)

    def package(self, item):
        """What actually goes over the link for one memory."""
        rep = item["rep"]
        if rep == "chunk":
            data = open(self._path(item["id"], "full"), "rb").read()[item["offset"]: item["offset"] + item["n"]]
            return {"id": item["id"], "node": self.cfg.node_id, "hlc": self._tick(), "rep": "chunk",
                    "offset": item["offset"], "total": item["full_size"], "image": base64.b64encode(data).decode()}
        body = {
            "id": item["id"], "node": self.cfg.node_id, "hlc": self._tick(), "captured_at": item["captured_at"],
            "rep": rep, "kind": item.get("kind", "observation"), "ref": item.get("ref"),
            "tags": item["tags"], "label": item["label"],
            "importance": item.get("importance", 0.0), "emb": quantize(item["vec"]),
        }
        if item.get("pos"):
            body.update(pos=item["pos"], geo=item["geo"])
        if rep == "text":
            for k in ("text", "evidence", "site", "action", "reason", "qid", "question", "answer", "source"):
                if k in item:
                    body[k] = item[k]
            # evidence the gate never sends because another robot already delivered the same scene: tell ground which
            # fleet item it duplicates, so the conclusion lands on that site (and can meet a contradicting one there)
            alias = {e: a for e in item.get("evidence", []) if (a := self._fleet_alias(e))}
            if alias:
                body["evidence_alias"] = alias
        else:
            body["caption"] = item.get("caption")
            body["image"] = base64.b64encode(open(self._path(item["id"], "thumb"), "rb").read()).decode()
        return body

    def mark_sent(self, item):
        state = "state"
        if item["rep"] == "chunk":
            sent = item["offset"] + item["n"]
            done = sent >= item["full_size"]
            upd = {"full_sent": sent, **({state: "full", "requested_full": 0} if done else {})}
            if done:
                self.store.event("downlink", f"{item['id'][:8]} full resolution delivered "
                                             f"({item['full_size']} B streamed across windows)")
        else:
            upd = {state: "sent" if item["rep"] == "text" else "thumb"}
        self.store.set_payload([item["id"]], upd)

    # ------------------------------------------------------------ bounded memory

    def storage_bytes(self):
        return sum(os.path.getsize(os.path.join(self.store.img_dir, f)) for f in os.listdir(self.store.img_dir))

    def enforce_budget(self):
        """Tier raw evidence down when storage is over budget: local originals, then full-res of delivered low-value
        observations, then thumbnails already delivered. Embeddings and text memories always stay."""
        budget = self.cfg.storage_mb * 1e6
        used = self.storage_bytes()
        if used <= budget:
            return 0
        recs = [r for r in self.store.scroll(Filter(must=_kind_is("observation")))]
        recs.sort(key=lambda r: (r.payload.get("importance", 0), r.payload.get("influence", 0), r.payload["captured_at"]))
        freed = 0
        steps = (
            ("orig", lambda p: True, "thumb+full"),
            ("full", lambda p: p["state"] != "new" and p.get("importance", 0) < 0.5 and not p.get("requested_full"), "thumb"),
            ("thumb", lambda p: p["state"] != "new", "embedding"),
        )
        for kind, ok, tier in steps:
            for r in recs:
                if used - freed <= budget:
                    break
                path = self._path(str(r.id), kind)
                if ok(r.payload) and os.path.exists(path):
                    freed += os.path.getsize(path)
                    os.remove(path)
                    upd = {"tier": tier}
                    if kind == "full":
                        upd["full_size"] = 0
                    self.store.set_payload([str(r.id)], upd)
        if freed:
            self.store.event("memory", f"storage over {self.cfg.storage_mb:.0f} MB budget: freed {freed / 1e6:.1f} MB of raw "
                                       f"evidence; embeddings and insights keep the meaning")
        return freed

    def tiers(self):
        counts = {}
        for r in self.store.scroll(Filter(must=_kind_is("observation"))):
            counts[r.payload.get("tier", "full")] = counts.get(r.payload.get("tier", "full"), 0) + 1
        return {"tiers": counts, "storage_mb": round(self.storage_bytes() / 1e6, 2), "budget_mb": self.cfg.storage_mb}

    # ------------------------------------------------------------ commands from ground

    def apply_fleet(self, items):
        for it in items:
            self.store.upsert(it["id"], list(map(float, dequantize(it["emb"]))),
                              {"node": it["node"], "tags": it.get("tags", []), "kind": it.get("kind", "observation")},
                              shard="fleet")

    def request_full(self, pid):
        rec = self.store.get([pid])
        if not rec:
            return "unknown"
        p = rec[0].payload
        if not p.get("full_size") or not os.path.exists(self._path(pid, "full")):
            self.store.event("memory", f"full-resolution request for {pid[:8]}: evicted under the storage budget; "
                                       f"only the thumbnail and embedding remain")
            return "evicted"
        self.store.set_payload([pid], {"requested_full": 1})
        self.store.event("request", f"ground requested full resolution of {pid[:8]} "
                                    f"({p['full_size'] // 1000} KB, streamed in chunks, demand first)")
        return "queued"

    def relabel(self, pid, label, by, remote_hlc):
        rec = self.store.get([pid])
        if not rec:
            return False
        self.store.set_meta("hlc", self.hlc.observe(remote_hlc))
        p = rec[0].payload
        history = p.get("label_history", []) + [{"label": p["label"], "superseded_by": label, "by": by, "hlc": remote_hlc}]
        self.store.set_payload([pid], {"label": label, "label_history": history})
        self.store.event("relabel", f"label revised {p['label']!r} -> {label!r} by {by}")
        return True

    def discard(self, pid, remote_hlc):
        self.store.set_meta("hlc", self.hlc.observe(remote_hlc))
        self.store.delete([pid])
        for kind in ("orig", "thumb", "full"):
            try:
                os.remove(self._path(pid, kind))
            except FileNotFoundError:
                pass
        self.store.event("discard", f"{pid[:8]} discarded by ground (tombstone); memory freed")
