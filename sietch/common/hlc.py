"""Hybrid logical clock. Timestamps are strings that sort lexicographically in causal order.

Format: "<wall_ms:013d>.<counter:05d>.<node>". Wall-clock skew between nodes can't reorder causally
related events: observe() always moves this clock past any timestamp it has seen.
"""

import threading
import time


def parse(ts):
    ms, counter, node = ts.split(".", 2)
    return int(ms), int(counter), node


def fmt(ms, counter, node):
    return f"{ms:013d}.{counter:05d}.{node}"


class HLC:
    def __init__(self, node, clock=None, last=None):
        self.node = node
        self._clock = clock or (lambda: int(time.time() * 1000))
        self._ms, self._counter = (parse(last)[:2] if last else (0, 0))
        self._lock = threading.Lock()

    def now(self):
        """Timestamp a local event."""
        with self._lock:
            wall = self._clock()
            if wall > self._ms:
                self._ms, self._counter = wall, 0
            else:
                self._counter += 1
            return fmt(self._ms, self._counter, self.node)

    def observe(self, remote):
        """Merge a timestamp received from another node, then timestamp the receive event."""
        r_ms, r_counter, _ = parse(remote)
        with self._lock:
            wall = self._clock()
            ms = max(wall, self._ms, r_ms)
            if ms == self._ms == r_ms:
                counter = max(self._counter, r_counter) + 1
            elif ms == self._ms:
                counter = self._counter + 1
            elif ms == r_ms:
                counter = r_counter + 1
            else:
                counter = 0
            self._ms, self._counter = ms, counter
            return fmt(ms, counter, self.node)
