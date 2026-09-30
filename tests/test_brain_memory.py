"""Memory kinds, the brain's tool loop (scripted fake LLM, no Ollama needed), supersede and bounded memory."""

import glob
import os
import tempfile
import time

import pytest

from sietch.common.config import NodeConfig
from sietch.node.brain import Brain
from sietch.node.memory import NodeMemory, ref
from sietch.node.store import Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS = os.path.join(ROOT, "spikes", "photos", "commons")


@pytest.fixture()
def robot(perception, monkeypatch):
    d = tempfile.mkdtemp(prefix="sietch_brain_")
    monkeypatch.setenv("SIETCH_BRAIN", "fake-model")
    cfg = NodeConfig(node_id="test-bot", data_dir=d)
    store = Store(d, perception.dim)
    mem = NodeMemory(cfg, perception, store)
    mem.set_intent("evidence of past water")
    ids = {}
    for cat in ("water", "layered_rock", "laptop"):
        for p in sorted(glob.glob(os.path.join(PHOTOS, cat, "*.jpg")))[:2]:
            ids.setdefault(cat, []).append(mem.ingest(open(p, "rb").read(), source=cat)["id"])
    brain = Brain.__new__(Brain)  # no worker thread; we drive it synchronously
    Brain.__init__(brain, cfg, mem, store)
    yield mem, brain, ids, store
    store.close()


def scripted(brain, replies):
    calls = []

    def fake_chat(messages, tools=None, stream=False, on_token=None):
        calls.append({"messages": messages, "tools": bool(tools)})
        reply = replies.pop(0)
        if stream and on_token:
            on_token(reply["content"])
        return reply
    brain._chat = fake_chat
    return calls


def tool(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_examine_writes_insight_with_evidence_and_acts(robot):
    mem, brain, ids, store = robot
    target, rock = ids["water"][0], ids["layered_rock"][0]
    calls = scripted(brain, [
        {"role": "assistant", "content": "Cracked mud beside a shallow puddle on a dirt road."},
        tool("recall", query="layered rock water"),
        tool("remember", insight=f"Mud cracks here and layered rock at [{ref('observation', rock)}] suggest standing water once pooled.",
             importance=0.9),
        tool("act", action="flag_for_ground", reason="strong evidence of past water"),
    ])
    brain._set(task={"kind": "examine", "started": time.time()})
    brain._examine(target, why="test")
    assert calls[0]["messages"][0].get("images"), "the brain must look at the frame first"
    obs = store.get([target])[0].payload
    assert obs["examined"] == 1 and "puddle" in obs["caption"]
    assert obs["importance"] == 1.0, "flag_for_ground raises the observation's importance"
    insight = mem.texts(("insight",))[0]
    assert target in insight["evidence"] and rock in insight["evidence"]
    decision = mem.texts(("decision",))[0]
    assert decision["action"] == "flag_for_ground"
    steps = [s["step"] for s in brain.state["trace"]]
    assert steps[:2] == ["wake", "see"] and "remember" in steps and "act" in steps


def test_conclusions_travel_before_pictures(robot):
    mem, brain, ids, store = robot
    ins = mem.add_text("insight", "Standing water once pooled in this basin.", importance=0.9, evidence=[ids["water"][0]])
    queue = mem.rank()
    assert queue[0]["id"] == ins["id"], [q.get("kind") for q in queue[:3]]
    sel, _ = mem.plan_window(8000)
    assert sel[0]["rep"] == "text" and sel[0]["cost"] < 3000
    pkg = mem.package(sel[0])
    assert "image" not in pkg and pkg["text"].startswith("Standing water") and pkg["evidence"] == [ids["water"][0]]


def test_ask_answers_offline_with_citations(robot):
    mem, brain, ids, store = robot
    ins = mem.add_text("insight", "Layered rock strata indicate past water deposition.", importance=0.8,
                       evidence=[ids["layered_rock"][0]])
    scripted(brain, [{"role": "assistant", "content": f"Layered strata [{ins['ref']}] point to past water."}])
    brain._set(task={"kind": "ask", "started": time.time()})
    brain._ask("What suggests past water?", qid="q1", source="ground")
    a = brain.state["answers"][0]
    assert ins["id"] in [c["id"] for c in a["cited"]]
    answer_mem = mem.texts(("answer",))[0]
    assert answer_mem["qid"] == "q1" and answer_mem["importance"] == 1.0


def test_superseded_conclusion_is_not_recalled(robot):
    mem, brain, ids, store = robot
    wrong = mem.add_text("insight", "This basin is completely dry, no water ever.", importance=0.7)
    assert any(r["id"] == wrong["id"] for r in mem.recall("dry basin water")["insights"])
    mem.supersede(wrong["id"], "mission control", "Standing water once pooled here.", "9999999999999.00000.ground")
    assert not any(r["id"] == wrong["id"] for r in mem.recall("dry basin water")["insights"])


def test_bounded_memory_tiers_down_evidence_not_meaning(robot):
    mem, brain, ids, store = robot
    for pid in ids["laptop"] + ids["water"]:
        store.set_payload([pid], {"state": "thumb"})
    ins = mem.add_text("insight", "Laptops are calibration images.", importance=0.2, evidence=ids["laptop"])
    mem.cfg.storage_mb = 0.05  # force eviction
    freed = mem.enforce_budget()
    assert freed > 0
    tiers = mem.tiers()["tiers"]
    assert tiers.get("embedding", 0) >= 1, tiers
    assert store.get([ins["id"]]), "text memories are never evicted"
    assert mem.recall("laptop")["results"], "embeddings still answer recall after raw evidence is gone"


def test_reflex_only_brain_answers_from_recall(robot, monkeypatch):
    mem, brain, ids, store = robot
    brain.enabled = False
    brain.answer_without_brain("where is the water?", qid="q2", source="ground")
    ans = mem.texts(("answer",))[0]
    assert ans["qid"] == "q2" and "reflex-only" in ans["answer"]


def test_invented_citations_are_stripped(robot):
    mem, brain, ids, _ = robot
    real = ref("observation", ids["water"][0])
    text = f"Standing water here [{real}] and layered rock at [obs-aa11bb22] suggest a basin."
    assert mem.strip_unknown_refs(text) == f"Standing water here [{real}] and layered rock at suggest a basin."
    assert mem.resolve_refs(mem.strip_unknown_refs(text)) == [ids["water"][0]]
    assert mem.strip_unknown_refs("Mud cracks [obs-<this id>] mean water.") == "Mud cracks mean water."
    assert mem.strip_unknown_refs("Rock [<its id>] too.") == "Rock too."
    assert mem.strip_unknown_refs("Rock together with [its id] suggests water.") == "Rock together with suggests water."
