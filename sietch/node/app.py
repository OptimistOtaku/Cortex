"""One SIETCH robot per process (see sietch/node/server.py for the app itself).

Run: PYTHONUTF8=1 SIETCH_NODE=rover-1 uvicorn sietch.node.app:app --port 8101
     SIETCH_BRAIN=off for a reflex-only robot (no local LLM).
"""

from sietch.common.runtime import load_env, watch_parent

load_env()
watch_parent()

from sietch.common.config import NodeConfig  # noqa: E402
from sietch.node.perception import Perception  # noqa: E402
from sietch.node.server import create_app  # noqa: E402

cfg = NodeConfig()
app = create_app(cfg, Perception.shared(cfg.model, cfg.world))
