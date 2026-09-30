"""Spike 11 (SIETCH go/no-go): can on-device text->image search serve live mission intents?

Stages:
  fetch  - ~5 Wikimedia Commons images per category into spikes/photos/commons/<category>/ (first pass;
           real venue photos go in spikes/photos/own/<category>/ for the confirmation pass)
  eval   - embed every image with each model, run each intent, report hit@1 and precision@3
  faces  - YuNet face detection + blur on every image; report detections per category
  sheet  - contact sheet PNG so labels can be checked by eye

    python spikes/spike_11_images.py [fetch|eval|faces|sheet|all] [--own]
"""

import json
import os
import statistics
import sys
import time
import urllib.parse

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(HERE, "photos")
UA = {"User-Agent": "SIETCH-hackathon-spike/0.1 (research; contact via github)"}

CATEGORIES = {
    "red_object": "red bottle OR red apple OR red chair photograph",
    "bottle": "plastic water bottle photograph",
    "logo_sign": "shop sign",
    "glasses_person": "portrait man wearing eyeglasses",
    "person_no_glasses": "woman portrait",
    "layered_rock": "rock strata",
    "water": "puddle on road after rain",
    "plant": "potted plant indoor",
    "laptop": "laptop computer on desk",
}

# intent -> categories that count as relevant
INTENTS = {
    "something red": ["red_object"],
    "a bottle": ["bottle", "red_object"],
    "a company logo or brand sign": ["logo_sign"],
    "a person wearing glasses": ["glasses_person"],
    "layered sedimentary rock": ["layered_rock"],
    "evidence of water": ["water"],
    "a plant": ["plant"],
}

# Hand-checked relevance for the commons gallery (indices = contact-sheet order), verified by eye 2026-09-30.
# Category labels were noisy: logo_sign holds no logos, #8/#9 show no glasses, red flowers sit in plant.
QRELS_COMMONS = {
    "something red": {0, 33, 34, 35, 36, 37, 38, 39},
    "a bottle": {0, 1, 2, 3, 4, 35, 36},
    "a person wearing glasses": {5, 6, 7},
    "layered sedimentary rock": {15, 16, 17, 18, 19},
    "evidence of water": {40, 41, 42, 43, 44},
    "a plant": {30, 31, 32, 33, 34, 39},
    "a laptop": {10, 11, 12, 13, 14},
    "a portrait of a woman": {25, 26, 28, 29},
    "a ferris wheel at night": {22, 23},
}

MODELS = {
    "clip-ViT-B-32": ("Qdrant/clip-ViT-B-32-vision", "Qdrant/clip-ViT-B-32-text"),
    "siglip2-base": ("google/siglip2-base-patch16-224", "google/siglip2-base-patch16-224"),
}


def commons_search(query, n):
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6, "gsrlimit": n * 3, "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 640,
    }
    r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=UA, timeout=30)
    r.raise_for_status()
    pages = sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") in ("image/jpeg", "image/png") and ii.get("thumburl"):
            out.append((p["title"], ii["thumburl"]))
    return out[:n]


def stage_fetch(n=5):
    root = os.path.join(PHOTOS, "commons")
    manifest = []
    for cat, q in CATEGORIES.items():
        d = os.path.join(root, cat)
        os.makedirs(d, exist_ok=True)
        got = 0
        for title, url in commons_search(q, n):
            fn = os.path.join(d, f"{got:02d}.jpg")
            if not os.path.exists(fn):
                img = requests.get(url, headers=UA, timeout=60)
                if img.status_code != 200:
                    continue
                open(fn, "wb").write(img.content)
            manifest.append({"category": cat, "file": os.path.relpath(fn, HERE), "source": title})
            got += 1
        print(f"[fetch] {cat:18} {got} images")
    json.dump(manifest, open(os.path.join(root, "manifest.json"), "w"), indent=1)


def load_gallery(sub):
    root = os.path.join(PHOTOS, sub)
    items = []
    for cat in sorted(os.listdir(root)):
        d = os.path.join(root, cat)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((cat, os.path.join(d, f)))
    return items


def norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def stage_eval(sub):
    from fastembed import ImageEmbedding, TextEmbedding

    items = load_gallery(sub)
    cats = [c for c, _ in items]
    print(f"[eval] gallery={sub} n={len(items)} categories={len(set(cats))}")
    only = os.environ.get("SPIKE_MODELS")
    for name, (img_m, txt_m) in MODELS.items():
        if only and name not in only.split(","):
            continue
        t0 = time.perf_counter()
        im, tm = ImageEmbedding(model_name=img_m), TextEmbedding(model_name=txt_m)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        X = norm(list(im.embed([p for _, p in items], batch_size=8)))
        img_ms = (time.perf_counter() - t0) * 1000 / len(items)
        rows, t_ms = [], []
        qrels = QRELS_COMMONS if sub == "commons" else {i: {k for k, c in enumerate(cats) if c in r} for i, r in INTENTS.items()}
        for intent, rel in qrels.items():
            t0 = time.perf_counter()
            q = norm(list(tm.embed([intent]))[0])
            t_ms.append((time.perf_counter() - t0) * 1000)
            order = np.argsort(-(X @ q))
            top3 = [int(i) for i in order[:3]]
            hit1 = top3[0] in rel
            p3 = sum(i in rel for i in top3) / 3
            rows.append((intent, hit1, p3, [f"#{i}:{cats[i][:10]}" for i in top3]))
        h1 = sum(r[1] for r in rows) / len(rows)
        p3 = statistics.mean(r[2] for r in rows)
        print(f"\n== {name}: load {load_s:.0f}s, image embed {img_ms:.0f} ms/img, text embed {statistics.median(t_ms):.0f} ms")
        print(f"   hit@1 {h1:.0%}   precision@3 {p3:.0%}")
        for intent, hit, p, top3 in rows:
            print(f"   {'OK ' if hit else 'MISS'} p@3={p:.2f}  {intent!r:34} -> {top3}")


def stage_faces(sub):
    import cv2

    model = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
    det = cv2.FaceDetectorYN.create(model, "", (320, 320), 0.7)
    by_cat = {}
    out_dir = os.path.join(PHOTOS, "_blurred")
    os.makedirs(out_dir, exist_ok=True)
    ms = []
    for cat, path in load_gallery(sub):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        t0 = time.perf_counter()
        _, faces = det.detect(img)
        ms.append((time.perf_counter() - t0) * 1000)
        n = 0 if faces is None else len(faces)
        by_cat.setdefault(cat, []).append(n)
        if n:
            for f in faces:
                x, y, fw, fh = [int(v) for v in f[:4]]
                x, y = max(x, 0), max(y, 0)
                roi = img[y:y + fh, x:x + fw]
                if roi.size:
                    img[y:y + fh, x:x + fw] = cv2.GaussianBlur(roi, (51, 51), 30)
            cv2.imwrite(os.path.join(out_dir, f"{cat}_{os.path.basename(path)}"), img)
    print(f"[faces] YuNet median {statistics.median(ms):.1f} ms/img")
    for cat, counts in by_cat.items():
        print(f"   {cat:18} images with >=1 face: {sum(c > 0 for c in counts)}/{len(counts)}")


def stage_sheet(sub):
    from PIL import Image, ImageDraw

    items = load_gallery(sub)
    cols, tw, th = 9, 160, 120
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (cat, p) in enumerate(items):
        im = Image.open(p).convert("RGB")
        im.thumbnail((tw, th))
        x, y = (i % cols) * tw, (i // cols) * (th + 14)
        sheet.paste(im, (x, y))
        d.text((x + 2, y + th), f"{i}:{cat[:16]}", fill="black")
    out = os.path.join(PHOTOS, f"contact_{sub}.png")
    sheet.save(out)
    print(f"[sheet] {out}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    sub = "own" if "--own" in sys.argv else "commons"
    if stage in ("fetch", "all") and sub == "commons":
        stage_fetch()
    if stage in ("sheet", "all"):
        stage_sheet(sub)
    if stage in ("eval", "all"):
        stage_eval(sub)
    if stage in ("faces", "all"):
        stage_faces(sub)
