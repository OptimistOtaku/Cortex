"""Spike 08: reframe conflict detection for Jev. Clocks decide causality; Jev only judges semantic contradiction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_07_jev import ask  # noqa: E402

CONTRA = {
    "type": "noul",
    "instructions": "Do note A and note B make incompatible claims about the same property of the same asset, "
                    "so that both cannot be true at the same time?",
    "criteria": {"true": "They contradict each other about the same thing", "false": "Compatible, complementary, or about different things"},
}
PAIRS = [  # (A, B, expected contradiction)
    ("INV-17: E42 fault active, inverter offline", "INV-17: E42 cleared, inverter back online", True),
    ("T-3 battery bank at 48V, healthy", "T-3 battery bank at 41V, low-voltage alarm active", True),
    ("Site 9 gate code is 4471", "Site 9 gate code is 8820", True),
    ("Pump P-2 running normally", "Pump P-2 not running, no power at the panel", True),
    ("INV-17: E42 fault active", "INV-17: fan filter is dusty", False),
    ("T-3 battery bank at 48V", "T-4 battery bank at 41V, alarm active", False),
    ("Pump P-2 running normally", "Pump P-2 running, slight vibration noted", False),
    ("Site 9 gate code is 4471", "Site 9 has a dog on premises", False),
]

if __name__ == "__main__":
    ok = 0
    for a, b, exp in PAIRS:
        r, ms = ask({"A": a, "B": b}, {"contradiction": CONTRA})
        p = r["answers"]["contradiction"]["noul"]
        good = (p >= 0.5) == exp
        ok += good
        print(f"p={p:.2f} expected={exp!s:5} {'OK' if good else 'WRONG'} {ms:.0f}ms | {a[:34]!r} vs {b[:40]!r}")
    print(f"contradiction accuracy {ok}/{len(PAIRS)}")
