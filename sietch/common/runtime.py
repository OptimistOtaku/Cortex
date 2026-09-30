"""Process plumbing shared by rovers and ground: .env loading and a parent watchdog."""

import os
import threading
import time

from sietch.common.config import ROOT


def load_env(path=None):
    """KEY=VALUE lines from the repo's .env (git-ignored) into os.environ, without overriding what is set."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _alive(pid):
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def watch_parent():
    """Exit when the launcher dies, so fleet processes never outlive run_fleet.py and hold RAM (seen 2026-09-30)."""
    pid = int(os.environ.get("SIETCH_PARENT_PID", "0"))
    if not pid:
        return

    def loop():
        while True:
            time.sleep(3)
            if not _alive(pid):
                os._exit(0)

    threading.Thread(target=loop, daemon=True, name="parent-watchdog").start()
