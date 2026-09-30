"""Spike 13: does the robots' embedding model recognise Mars geology? (real NASA imagery, sietch/sim/fetch_assets.py)

Runs against a live host (python run_sim.py --fresh; model per --model): feeds every Mars image to scout-1 as a
capture (centre crop, as the simulated camera does), then asks text queries a mission would ask and checks the top
hits. Reports hit@1, hit@3 and the zero-shot Mars tags per image.

    python spikes/spike_13_mars.py
"""

import io
import json
import os

import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "sietch", "sim", "static", "assets", "mars")
NODE = "http://127.0.0.1:8102"
QUERIES = {
    "mud cracks": {"mud_cracks", "mud_cracks_close"},
    "layered sedimentary rock": {"delta_scarp", "kodiak"},
    "rounded pebbles from an ancient stream": {"streambed_pebbles"},
    "white mineral veins in rock": {"veins"},
    "rock with dark spots": {"leopard_spots"},
    "a meteorite": {"meteorite"},
    "a sand dune": {"dune", "aerial_ripples"},
    "a dust devil": {"dust_devil"},
    "sunset": {"sunset"},
    "aerial view of sand ripples": {"aerial_ripples"},
    "spacecraft debris": {"aerial_backshell"},
    "small round spheres": {"spherules"},
    "evidence of past water": {"mud_cracks", "mud_cracks_close", "streambed_pebbles", "veins", "delta_scarp", "kodiak", "spherules"},
}


def crop(path, keep=0.8):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    cw, ch = int(w * keep), int(h * keep)
    im = im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main():
    catalog = json.load(open(os.path.join(ASSETS, "catalog.json")))
    ids = {}
    for key in catalog:
        r = requests.post(f"{NODE}/api/capture", data=crop(os.path.join(ASSETS, f"{key}.jpg")),
                          headers={"X-Source": key}, timeout=60).json()
        ids[r["id"]] = key
        print(f"{key:18} tags {r['tags']}")
    h1 = h3 = 0
    for q, want in QUERIES.items():
        res = requests.get(f"{NODE}/api/search", params={"q": q, "k": 3}, timeout=30).json()["results"]
        got = [ids.get(r["id"], "?") for r in res]
        h1 += got[0] in want
        h3 += bool(set(got) & want)
        print(f"{'OK ' if got[0] in want else 'MISS'} {q:40} -> {got}")
    print(f"\nhit@1 {h1}/{len(QUERIES)}  hit@3 {h3}/{len(QUERIES)}")


if __name__ == "__main__":
    main()
