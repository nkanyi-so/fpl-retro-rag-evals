"""Regenerate the two README findings charts from the canonical decision_quality.py output.

Numbers are transcribed from the canonical `python src/evals/decision_quality.py`
run (n=8, K=5). Re-run that eval and update the literals here if the corpus changes.
Outputs: assets/calibration.png, assets/divergence_wins.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
INK = "#1d2433"
MUTED = "#6b7280"
RAG_C = "#2563eb"   # blue
CROWD_C = "#9ca3af"  # grey
BUST_C = "#dc2626"  # red

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#d1d5db",
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": MUTED,
})


def calibration_chart() -> None:
    """Chart 1 — confidence calibration: mean normalized capture per bucket.
    The inversion: high (0.40) sits BELOW medium (0.43). The high bucket's two
    template-agreement busts (GW9, GW11) are what drag it under; GW13 is the
    medium-bucket bust. (floor=0.0, ceiling=1.0)."""
    buckets = ["medium\n(0.50–0.79)", "high\n(≥0.80)"]
    capture = [0.43, 0.40]
    colors = ["#3b82f6", "#1e3a8a"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(buckets, capture, width=0.55, color=colors, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=13, color=INK, fontweight="bold")

    # the inversion arrow
    ax.annotate(
        "higher confidence,\nLOWER capture — inverted",
        xy=(1, 0.40), xytext=(0.62, 0.62),
        fontsize=10, color=BUST_C, ha="center",
        arrowprops=dict(arrowstyle="->", color=BUST_C, lw=1.4),
    )

    # bust annotations
    ax.annotate("GW9: 2 pts (conf 0.82)\nGW11: 4 pts (conf 0.89)\nconfident template busts",
                xy=(1, 0.40), xytext=(1.02, 0.20), fontsize=9, color=BUST_C, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=BUST_C, lw=0.8))
    ax.annotate("GW13: 2 pts (conf 0.64)\nmedium bust", xy=(0, 0.43), xytext=(-0.02, 0.20),
                fontsize=9, color=MUTED, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f9fafb", ec="#d1d5db", lw=0.8))

    ax.set_ylim(0, 0.72)
    ax.set_ylabel("mean normalized capture\n(random floor = 0.0, perfect hindsight = 1.0)")
    ax.set_title("Confidence is not calibrated to outcome  (n=8)",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eef0f3", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=160)
    plt.close(fig)


def divergence_chart() -> None:
    """Chart 2 — the divergence-only wins: on the three GWs where RAG left the
    crowd, RAG points vs crowd (most-captained) points. +5 / +4 / +10 = +19."""
    gws = ["GW1", "GW5", "GW15"]
    rag_pts = [13, 9, 12]
    crowd_pts = [8, 5, 2]
    rag_lbl = ["Haaland", "Haaland", "Foden"]
    crowd_lbl = ["Salah", "Salah", "Haaland"]
    deltas = [r - c for r, c in zip(rag_pts, crowd_pts)]

    x = range(len(gws))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], rag_pts, w, label="RAG pick", color=RAG_C, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], crowd_pts, w, label="Crowd (most-captained)",
                color=CROWD_C, zorder=3)

    for i in x:
        ax.text(i - w / 2, rag_pts[i] + 0.3, f"{rag_lbl[i]}\n{rag_pts[i]}",
                ha="center", va="bottom", fontsize=9, color=RAG_C, fontweight="bold")
        ax.text(i + w / 2, crowd_pts[i] + 0.3, f"{crowd_lbl[i]}\n{crowd_pts[i]}",
                ha="center", va="bottom", fontsize=9, color=MUTED)
        ax.text(i, max(rag_pts[i], crowd_pts[i]) + 2.4, f"+{deltas[i]}",
                ha="center", fontsize=12, color="#16a34a", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{g}\n(medium conf)" for g in gws])
    ax.set_ylim(0, 18)
    ax.set_ylabel("captain points (real 2025–26 outcome)")
    ax.set_title("Divergence-only wins: +19 over 3 GWs  (all medium-confidence)",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eef0f3", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "divergence_wins.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    calibration_chart()
    divergence_chart()
    print("wrote calibration.png, divergence_wins.png")
