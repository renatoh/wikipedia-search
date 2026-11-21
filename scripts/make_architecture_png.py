"""
Render the runtime architecture diagram to a PNG using matplotlib.
Run: python make_architecture_png.py  ->  docs/architecture.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent.parent / "docs" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# palette
C_ENTRY = "#DCE9F7"
C_ENTRY_E = "#5B8DBE"
C_AGENT = "#FCE3C2"
C_AGENT_E = "#D89A3C"
C_LLM = "#D9EFD6"
C_LLM_E = "#5FA85A"
C_OS = "#E5DBF2"
C_OS_E = "#8B63C0"
INK = "#1c1c1c"

fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=12, weight="bold", subtext=None):
    """x,y = center."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.8, facecolor=fc, edgecolor=ec,
        )
    )
    if subtext:
        ax.text(x, y + h * 0.16, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=INK)
        ax.text(x, y - h * 0.22, subtext, ha="center", va="center",
                fontsize=fs - 3, color="#444")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=INK)


def arrow(x1, y1, x2, y2, style="-|>", color=INK, lw=1.8, rad=0.0, label=None,
          lx=None, ly=None, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=16,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}", linestyle=ls,
        )
    )
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2,
                ly if ly is not None else (y1 + y2) / 2,
                label, ha="center", va="center", fontsize=8.5,
                color="#333", style="italic")


# ---- self-contained boundary ----
ax.add_patch(
    FancyBboxPatch(
        (3, 3), 94, 84,
        boxstyle="round,pad=0.6,rounding_size=2",
        linewidth=1.6, facecolor="none", edgecolor="#B0483C",
        linestyle=(0, (6, 4)),
    )
)
ax.text(50, 91.5, "SELF-CONTAINED AGENTIC SEARCH",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#B0483C")

# ---- entry points ----
box(28, 80, 22, 8, "Web UI", C_ENTRY, C_ENTRY_E)
box(56, 80, 22, 8, "CLI", C_ENTRY, C_ENTRY_E)

# ---- agent ----
box(42, 60, 32, 12, "Agent", C_AGENT, C_AGENT_E, fs=15, subtext="tool loop")

# ---- LLM (left) ----
box(20, 34, 34, 14, "LLM", C_LLM, C_LLM_E, fs=14,
    subtext="Qwen 3.6  (35B-A3B, via Ollama)")

# ---- OpenSearch (right) ----
ax.add_patch(
    FancyBboxPatch(
        (56, 27), 38, 15,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.8, facecolor=C_OS, edgecolor=C_OS_E,
    )
)
ax.text(75, 38, "OpenSearch", ha="center", va="center",
        fontsize=14, fontweight="bold", color=INK)
ax.text(75, 33.6, "Wikipedia index (keyword + vector)",
        ha="center", va="center", fontsize=9.5, color="#444")
ax.text(75, 30.4, "hybrid search · RRF fusion",
        ha="center", va="center", fontsize=9.5, color="#444")

# ---- embedding model ----
box(75, 10, 34, 9, "Embedding model", C_LLM, C_LLM_E, fs=12.5,
    subtext="Qwen3-Embedding-4B")

# ================= arrows (all black) =================
# entry -> agent
arrow(28, 75.5, 38, 66.5)
arrow(56, 75.5, 46, 66.5)

# agent <-> LLM  (ask / answer) — two clearly separated lanes
arrow(34, 54, 16, 41.5, rad=0.28, label="asks what\nto search", lx=16, ly=50)
arrow(24, 41.5, 40, 54, rad=0.28, label="answer", lx=41, ly=49)

# agent -> OpenSearch
arrow(53, 57, 66, 42.5, rad=-0.12, label="run search /\nfetch articles", lx=68, ly=52)

# OpenSearch <-> embedding model
arrow(70, 26.7, 70, 14.7, label="query\ntext", lx=64.5, ly=20.5)
arrow(80, 14.7, 80, 26.7, label="query\nvector", lx=85.5, ly=20.5)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
