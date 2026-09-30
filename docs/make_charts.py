"""README charts from measured numbers (sources: findings_results.md, 2026-09-29/30). pip install matplotlib."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
BG, FG, MUTED, GATE, BRAIN, GOOD, HELD = "#140d0b", "#f3e7dc", "#b59d8a", "#f2a33a", "#5cc8e6", "#3fbf7f", "#6f9cf0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


# 1. why conclusions go first: bytes on the wire per memory
fig, ax = plt.subplots(figsize=(8, 3.4))
kinds = ["Conclusion\n(insight)", "Decision", "Picture\n(thumbnail)", "Picture\n(full resolution)"]
kb = [1.8, 1.7, 8.1, 305]
bars = ax.barh(kinds, kb, color=[BRAIN, BRAIN, GATE, GATE])
ax.set_xscale("log")
ax.set_xlabel("KB on the wire (log scale)")
ax.set_title("What one memory costs over a 30 KB orbiter pass")
for b, v in zip(bars, kb):
    ax.text(v * 1.08, b.get_y() + b.get_height() / 2, f"{v:g} KB", va="center", color=FG)
ax.axvline(30, color=MUTED, ls="--", lw=1)
ax.text(30 * 1.05, -0.55, "one pass = 30 KB", color=MUTED, fontsize=9)
ax.invert_yaxis()
save(fig, "chart_bytes.png")

# 2. one real pass (Mars demo, take 3): what the gate sent vs what stayed on board
fig, ax = plt.subplots(figsize=(8, 2.6))
parts = [("2 conclusions", 3.6, BRAIN), ("2 decisions", 3.5, "#3a8fa6"), ("2 best pictures", 16.3, GATE),
         ("unused", 6.6, "#3a2d27")]
left = 0
for label, v, c in parts:
    ax.barh([0], [v], left=left, color=c)
    ax.text(left + v / 2, 0, f"{label}\n{v:.1f} KB", ha="center", va="center", fontsize=9,
            color=BG if c in (BRAIN, GATE) else FG)
    left += v
ax.set_xlim(0, 30)
ax.set_yticks([])
ax.set_xlabel("KB of the 30 KB pass")
ax.set_title("One orbiter pass, live: meaning first, 14 more memories wait on board")
save(fig, "chart_pass.png")

# 3. offline recall latency on the rover: where the time goes
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.barh(["Mars demo\n(SigLIP2, 768-d)", "Earth gallery\n(CLIP, 512-d)"], [206.5, 12.5], color=MUTED, label="embed the question")
ax.barh(["Mars demo\n(SigLIP2, 768-d)", "Earth gallery\n(CLIP, 512-d)"], [2.5, 0.7], left=[206.5, 12.5], color=GOOD,
        label="Qdrant Edge hybrid search")
ax.text(213, 0, "Qdrant Edge: 2.5 ms", va="center", color=GOOD)
ax.text(17, 1, "Qdrant Edge: 0.7 ms", va="center", color=GOOD)
ax.set_xlim(0, 300)
ax.set_xlabel("milliseconds, airplane mode (no network)")
ax.set_title("Offline recall: the vector search itself is milliseconds")
ax.legend(frameon=False, loc="lower right", fontsize=9)
save(fig, "chart_recall.png")

# 4. reflex vs deliberation: the LLM wakes only when the Formula reflex says so
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.bar(["frames seen", "frames that woke\nthe LLM brain"], [20, 4], color=[MUTED, BRAIN], width=0.5)
ax.text(0, 20.4, "20 · reflex: ms each", ha="center")
ax.text(1, 4.4, "4 · deliberation: ~10-40 s each", ha="center")
ax.set_ylim(0, 24)
ax.set_ylabel("frames")
ax.set_title("Reflex vs deliberation (e2e test): thinking is spent where it matters")
save(fig, "chart_reflex.png")

# 5. one laptop: shared model host vs one process per robot
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.barh(["3 processes\n(CLIP, measured)", "1 shared host\n(SigLIP2, measured)"], [2.95, 2.28], color=[MUTED, GOOD])
ax.text(3.0, 0, "2.95 GB with the lighter model", va="center")
ax.text(2.33, 1, "2.28 GB with the better model", va="center")
ax.set_xlim(0, 4.6)
ax.set_xlabel("GB private memory: ground + rover + scout")
ax.set_title("Everything on one 7.3 GB laptop: one embedding model shared by all robots")
save(fig, "chart_ram.png")
print("charts written to", OUT)
