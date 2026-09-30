"""Spatial memory: geotagged captures, Qdrant Edge geo-radius recall, and "where" in the brain's context."""

import glob
import os
import tempfile

import pytest

from sietch.common.config import NodeConfig
from sietch.common.geo import parse_pos, to_geo, where
from sietch.node.memory import NodeMemory
from sietch.node.store import Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS = os.path.join(ROOT, "spikes", "photos", "commons")


@pytest.fixture()
def mem(perception):
    d = tempfile.mkdtemp(prefix="sietch_geo_")
    store = Store(d, perception.dim)
    m = NodeMemory(NodeConfig(node_id="geo-bot", data_dir=d), perception, store)
    yield m
    store.close()


def _img(cat, i=0):
    return open(sorted(glob.glob(os.path.join(PHOTOS, cat, "*.jpg")))[i], "rb").read()


def test_where_and_parse():
    assert parse_pos("12.5, -40") == {"x": 12.5, "z": -40.0}
    assert parse_pos(None) is None and parse_pos("nope") is None
    assert where({"x": 0, "z": -38}, {"x": 0, "z": 0}) == "38 m N"
    assert where({"x": 30, "z": 30}, {"x": 0, "z": 0}) == "42 m SE"
    assert where({"x": 1, "z": 1}, {"x": 0, "z": 0}) == "right here"
    a, b = to_geo(0, 0), to_geo(0, -1000)
    assert abs((b["lat"] - a["lat"]) * 111_195 - 1000) < 1  # metres map 1:1 onto Qdrant's geo distance


def test_capture_is_geotagged_and_recalled_by_place(mem):
    water = mem.ingest(_img("water"), source="water", pos={"x": 0.0, "z": 0.0})
    rock = mem.ingest(_img("layered_rock"), source="layered_rock", pos={"x": 200.0, "z": 0.0})
    assert water["geo"] and water["pos"] == {"x": 0.0, "z": 0.0}
    here = mem.near({"x": 5.0, "z": 0.0}, radius_m=30)
    assert [r["id"] for r in here] == [water["id"]]
    assert here[0]["where"] == "5 m W"
    near_rock = mem.recall("rock", k=5, near=({"x": 200.0, "z": 0.0}, 30))
    assert [r["id"] for r in near_rock["results"]] == [rock["id"]], "the geo filter keeps only what is near"
    ins = mem.add_text("insight", "Standing water here.", importance=0.9, evidence=[water["id"]])
    assert ins["pos"] == water["pos"], "a conclusion lives where its evidence was seen"
    pkg_fields = mem.package({**mem.rank()[0], "rep": "text"})
    assert "geo" in pkg_fields and "pos" in pkg_fields
