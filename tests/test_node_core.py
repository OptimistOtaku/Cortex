"""Node core on the spike gallery: ingest, offline search, retasking, window plan, replay."""

import glob
import os
import subprocess
import sys
import tempfile

import pytest

from sietch.common.config import NodeConfig
from sietch.node.memory import NodeMemory
from sietch.node.store import Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY = sorted(glob.glob(os.path.join(ROOT, "spikes", "photos", "commons", "*", "*.jpg")))


@pytest.fixture(scope="module")
def node(perception):
    d = tempfile.mkdtemp(prefix="sietch_")
    cfg = NodeConfig(node_id="test-rover", data_dir=d)
    store = Store(d, perception.dim)
    mem = NodeMemory(cfg, perception, store)
    for p in GALLERY:
        mem.ingest(open(p, "rb").read(), source=os.path.basename(os.path.dirname(p)))
    yield mem
    store.close()


def _by_source(node):
    return {r.payload["source"]: r.payload for r in node.store.scroll()}


def test_ingested_everything(node):
    assert node.store.count() == len(GALLERY) == 45


def test_offline_hybrid_search(node):
    res = node.search("a bottle")
    assert res["results"][0]["source"] in ("bottle", "red_object")
    assert res["ms"] < 500


def test_retasking_reorders_queue(node):
    node.set_intent("layered sedimentary rock")
    top = [r["source"] for r in node.rank()[:3]]
    node.set_intent("evidence of water")
    top_water = [r["source"] for r in node.rank()[:3]]
    assert top.count("layered_rock") >= 2, top
    assert top_water.count("water") >= 1, top_water
    assert top != top_water


def test_window_respects_budget_and_skips_duplicates(node):
    sel, skipped = node.plan_window(40_000)
    assert sum(s["cost"] for s in sel) <= 40_000 and sel
    assert any("near-duplicate" in why or "budget" in why for _, why in skipped), skipped[:3]
    thumbs = [s for s in sel if s["rep"] == "thumb"]
    assert all(float(a["vec"] @ b["vec"]) <= node.cfg.near_duplicate for i, a in enumerate(thumbs) for b in thumbs[i + 1:])


def test_replay_after_hard_exit(perception):
    """Write, then os._exit before any flush; reopening must replay our SQLite log."""
    d = tempfile.mkdtemp(prefix="sietch_crash_")
    code = f"""
import os, sys
sys.path.insert(0, {ROOT!r})
from sietch.common.config import NodeConfig
from sietch.node.store import Store
s = Store({d!r}, 8)
s._stop.set()  # stop the background flusher: simulate a crash before any group commit
for i in range(20):
    s.upsert("00000000-0000-0000-0000-0000000000%02d" % i, [1.0] * 8, {{"i": i}})
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    s = Store(d, 8)
    assert s.replayed == 20
    assert s.count() == 20
    s.close()
