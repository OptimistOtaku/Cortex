"""Settings from environment variables, so one codebase runs as any node or as ground control."""

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FastEmbed text<->image model pairs (validated in spikes/spike_11_images.py)
MODELS = {
    "siglip2": ("google/siglip2-base-patch16-224", "google/siglip2-base-patch16-224", 768),
    "clip": ("Qdrant/clip-ViT-B-32-vision", "Qdrant/clip-ViT-B-32-text", 512),
}


@dataclass
class NodeConfig:
    node_id: str = field(default_factory=lambda: os.environ.get("SIETCH_NODE", "rover-1"))
    port: int = field(default_factory=lambda: int(os.environ.get("SIETCH_PORT", "8101")))
    data_dir: str = field(default_factory=lambda: os.environ.get("SIETCH_DATA", ""))
    ground_url: str = field(default_factory=lambda: os.environ.get("SIETCH_GROUND", "http://127.0.0.1:8100"))
    model: str = field(default_factory=lambda: os.environ.get("SIETCH_MODEL", "siglip2"))
    world: str = field(default_factory=lambda: os.environ.get("SIETCH_WORLD", "earth"))  # earth | mars: tag vocabulary
    flush_interval_s: float = 0.25
    thumb_px: int = 192
    thumb_quality: int = 60
    full_px: int = 1024
    full_quality: int = 85
    # sync gate weights: value = w_intent*intent_sim + w_novelty*novelty + w_influence*influence, ranked per KB
    w_intent: float = 1.0
    w_novelty: float = 0.5
    w_influence: float = 0.3
    near_duplicate: float = 0.93
    w_importance: float = 1.0  # the brain's (or an operator's) judgement of how much ground needs a memory
    # brain: deliberate only when the reflex flags an observation (the LLM needs ~10 s per deliberation, spike_12)
    brain_model: str = field(default_factory=lambda: os.environ.get("SIETCH_BRAIN", "qwen3-vl:2b-instruct"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("SIETCH_OLLAMA", "http://127.0.0.1:11434"))
    brain_relevance: float = 0.75  # normalised mission relevance within the queue
    brain_novelty: float = 0.12
    # bounded memory: raw evidence is tiered down when image storage exceeds this; insights always stay
    storage_mb: float = field(default_factory=lambda: float(os.environ.get("SIETCH_STORAGE_MB", "200")))

    def __post_init__(self):
        if not self.data_dir:
            # outside OneDrive and short enough for Windows MAX_PATH
            self.data_dir = os.path.join(os.path.expanduser("~"), ".sietch", self.node_id)
        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def dim(self):
        return MODELS[self.model][2]
