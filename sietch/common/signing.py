"""HMAC-signed commands from mission control.

A spoofed "discard" or "retask" could wipe or hijack a robot, so every uplink command is signed with a key shared
by ground and the fleet (SIETCH_KEY) and a rover refuses anything that doesn't verify.
"""

import hashlib
import hmac
import json
import os

DEV_KEY = "sietch-dev-key-change-me"


def _key():
    return os.environ.get("SIETCH_KEY", DEV_KEY).encode()


def canonical(cmd):
    fields = {k: cmd[k] for k in ("id", "node", "kind", "body", "hlc")}
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def sign(cmd):
    return hmac.new(_key(), canonical(cmd).encode(), hashlib.sha256).hexdigest()


def verify(cmd):
    return hmac.compare_digest(str(cmd.get("sig", "")), sign(cmd))
