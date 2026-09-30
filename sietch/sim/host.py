"""Ground control, a rover and a scout drone in one process that shares one perception model.

For a single laptop: separate processes would each load the embedding model (~1.2-2 GB apiece). The robots stay
separate in every other way: their own Qdrant Edge shards, SQLite logs, brains, comms loops and HTTP ports, and
they reach ground only over HTTP in comms windows.

    python -m sietch.sim.host      (normally started by run_sim.py)
"""

import asyncio
import glob
import os

import uvicorn

from sietch.common.runtime import load_env, watch_parent

load_env()
watch_parent()

from sietch.common.config import NodeConfig  # noqa: E402
from sietch.node.perception import Perception  # noqa: E402
from sietch.node.server import create_app  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "assets", "mars")

ROBOTS = (("rover-1", 8101), ("scout-1", 8102))


async def serve(apps):
    servers = [uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")) for app, port in apps]
    await asyncio.gather(*(s.serve() for s in servers))


def main():
    model, world = os.environ.get("SIETCH_MODEL", "siglip2"), os.environ.get("SIETCH_WORLD", "mars")
    perception = Perception.shared(model, world)  # created before ground is imported, so ground reuses it
    if world == "mars":
        from PIL import Image

        perception.calibrate_tags([Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(ASSETS, "*.jpg")))])
    import sietch.ground.app as ground

    apps = [(ground.app, 8100)]
    for node, port in ROBOTS:
        apps.append((create_app(NodeConfig(node_id=node, port=port, model=model, world=world), perception), port))
    asyncio.run(serve(apps))


if __name__ == "__main__":
    main()
