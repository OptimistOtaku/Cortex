"""On-device perception: image/text embeddings in one space and zero-shot tags.

Everything here runs offline once models are cached. Run Python with PYTHONUTF8=1 on Windows:
fastembed reads SigLIP2's tokenizer config with the locale encoding otherwise (spike_11).
"""

import io
import threading

import numpy as np

from sietch.common.config import MODELS

# zero-shot tag vocabularies per world: tags label memories and feed the BM25 half of hybrid recall
TAG_VOCABS = {
    "earth": [
        "rock", "layered rock", "sand", "soil", "water", "puddle", "ice", "plant", "tree", "flower", "grass",
        "bottle", "cup", "can", "bag", "laptop", "phone", "keyboard", "screen", "book", "paper", "poster", "logo",
        "sign", "text", "person", "face", "hand", "shoe", "chair", "table", "door", "window", "wall", "floor",
        "light", "cable", "tool", "box", "car", "bicycle", "food", "fruit", "badge", "sticker", "stage", "crowd",
        "red object", "blue object", "green object", "yellow object", "white object", "black object",
        "metal", "wood", "glass", "plastic", "fabric",
    ],
    "mars": [
        "layered rock", "sediment layers", "mud cracks", "rounded pebbles", "conglomerate", "white mineral veins",
        "spotted rock", "meteorite", "shiny metal rock", "sand dune", "sand ripples", "dark sand", "dust devil",
        "dust", "gravel", "pebbles", "flat bedrock", "boulders", "rock outcrop", "cliff", "crater", "hills",
        "river delta", "dry riverbed", "light-toned rock", "cracked rock", "smooth rock", "porous volcanic rock",
        "drill hole", "wheel tracks", "rover hardware", "parachute", "spacecraft debris", "helicopter shadow",
        "ice", "frost", "red soil", "sky", "horizon", "sunset", "shadows",
    ],
}
TAG_TEMPLATE = {"earth": "a photo of a {}", "mars": "a photo of {}"}


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Perception:
    _shared = {}

    @classmethod
    def shared(cls, model="siglip2", world="earth"):
        """One model per process: a host running ground and several robots loads it once (~1.2-2 GB each)."""
        if (model, world) not in cls._shared:
            cls._shared[(model, world)] = cls(model, world)
        return cls._shared[(model, world)]

    @classmethod
    def existing(cls, model):
        return next((p for (m, _), p in cls._shared.items() if m == model), None)

    def __init__(self, model="siglip2", world="earth"):
        from fastembed import ImageEmbedding, TextEmbedding

        img_name, txt_name, self.dim = MODELS[model]
        self.model, self.world, self.vocab = model, world, TAG_VOCABS[world]
        self._img = ImageEmbedding(model_name=img_name)
        self._txt = TextEmbedding(model_name=txt_name)
        self._lock = threading.Lock()  # onnxruntime sessions are shared; keep calls serial
        self._tag_vecs = self.text_many([TAG_TEMPLATE[world].format(t) for t in self.vocab])
        self._tag_bias = np.zeros(len(self.vocab), dtype=np.float32)

    def image(self, pil):
        with self._lock:
            return _norm(list(self._img.embed([pil]))[0])

    def text(self, s):
        return self.text_many([s])[0]

    def text_many(self, texts):
        with self._lock:
            return _norm(list(self._txt.embed(texts)))

    def calibrate_tags(self, images):
        """Remove each tag's baseline similarity on typical images of this world. Without it a few generic tags
        ("crater", "light-toned rock") top every Mars image (hubness, spike_13)."""
        vecs = np.stack([self.image(im) for im in images])
        self._tag_bias = (vecs @ self._tag_vecs.T).mean(axis=0)

    def tags(self, vec, k=3):
        sims = self._tag_vecs @ vec
        order = np.argsort(-(sims - self._tag_bias))[:k]
        return [self.vocab[i] for i in order], [float(sims[i]) for i in order]


def jpeg(pil, max_px, quality):
    im = pil.convert("RGB")
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()
