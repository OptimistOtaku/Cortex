"""Fetch the real Mars imagery the simulated cameras show at science targets (NASA Image and Video Library).

NASA imagery is not copyrighted in the US; credits are kept in CREDITS.md and on the demo's end card.

    python -m sietch.sim.fetch_assets
"""

import io
import json
import os

import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "static", "assets", "mars")
API = "https://images-api.nasa.gov"

# key: (NASA id, camera that "takes" it in the sim, what it is)
CATALOG = {
    "delta_scarp": ("PIA24683", "mastcam", "Jezero delta scarp: layered sediments laid down by a river"),
    "kodiak": ("PIA24802", "mastcam", "Kodiak butte: delta remnant with inclined layers"),
    "mud_cracks": ("PIA21262", "mastcam", "Old Soaker: possible mud cracks, a lake bed that dried"),
    "mud_cracks_close": ("PIA21261", "mastcam", "Old Soaker close-up: polygonal cracks preserved in rock"),
    "veins": ("PIA19161", "mastcam", "Garden City: mineral veins where water flowed through fractures"),
    "streambed_pebbles": ("PIA16156", "mastcam", "Hottah: rounded pebbles from an ancient streambed"),
    "spherules": ("PIA05587", "mastcam", "Hematite 'berries': concretions grown in groundwater"),
    "leopard_spots": ("PIA26368", "mastcam", "Cheyava Falls: 'leopard spot' rock, a potential biosignature"),
    "meteorite": ("PIA21134", "mastcam", "Egg Rock: an iron-nickel meteorite"),
    "dune": ("PIA20283", "mastcam", "Namib dune: dark wind-blown sand"),
    "dust_devil": ("PIA25657", "navcam", "A dust devil crossing the crater floor"),
    "sunset": ("PIA19400", "mastcam", "Sunset in Gale crater: Mars sunsets are blue"),
    "aerial_ripples": ("PIA26242", "aerial", "From the air: sand ripples"),
    "aerial_first": ("PIA24593", "aerial", "From the air: first aerial colour image of Mars"),
    "aerial_backshell": ("PIA25219", "aerial", "From the air: the rover's discarded backshell"),
    "aerial_rover": ("PIA24625", "aerial", "From the air: the rover below the helicopter"),
}


def fetch(nasa_id):
    meta = requests.get(f"{API}/search", params={"nasa_id": nasa_id}, timeout=30).json()["collection"]["items"][0]["data"][0]
    files = requests.get(f"https://images-assets.nasa.gov/image/{nasa_id}/collection.json", timeout=30).json()
    for size in ("~large.jpg", "~medium.jpg", "~orig.jpg"):
        url = next((f for f in files if f.endswith(size)), None)
        if url:
            data = requests.get(url.replace("http://", "https://"), timeout=120).content
            return meta, Image.open(io.BytesIO(data)).convert("RGB")
    raise RuntimeError(f"no jpg for {nasa_id}")


def main():
    os.makedirs(OUT, exist_ok=True)
    catalog, credits = {}, ["# Mars imagery credits", "",
                            "Real mission imagery shown by the simulator's cameras at science targets. "
                            "NASA content is not copyrighted in the United States.", ""]
    for key, (nasa_id, camera, what) in CATALOG.items():
        meta, im = fetch(nasa_id)
        im.thumbnail((1280, 1280))
        im.save(os.path.join(OUT, f"{key}.jpg"), "JPEG", quality=88, optimize=True)
        credit = meta.get("secondary_creator") or "NASA/JPL-Caltech"
        catalog[key] = {"nasa_id": nasa_id, "camera": camera, "what": what, "title": meta.get("title"),
                        "credit": credit, "date": (meta.get("date_created") or "")[:10], "w": im.width, "h": im.height}
        credits.append(f"- `{key}.jpg`: {nasa_id} \"{meta.get('title')}\" ({credit}, {catalog[key]['date']})")
        print(f"{key:18} {nasa_id} {im.width}x{im.height}  {meta.get('title')}")
    json.dump(catalog, open(os.path.join(OUT, "catalog.json"), "w"), indent=1)
    open(os.path.join(OUT, "CREDITS.md"), "w", encoding="utf-8").write("\n".join(credits) + "\n")


if __name__ == "__main__":
    main()
