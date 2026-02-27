#!/usr/bin/env python3
"""Pipeline timeline: DFT->dpdata->train->test->LAMMPS with actual wall times."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "pipeline_timeline.png")

fig, ax = plt.subplots(figsize=(14, 5))

# Pipeline stages with real times from our runs
stages = [
    ("DFT\n(QE AIMD)", "2\u20134 hrs\n(100 frames, 32 atoms)", "#E53935", 0),
    ("dpdata\nconversion", "< 1 min", "#FB8C00", 1),
    ("DeePMD\ntraining", "~30 min (Ar)\n~2 hrs (water)", "#43A047", 2),
    ("dp test", "< 1 min", "#1E88E5", 3),
    ("LAMMPS\nvalidation", "~15 sec (Ar)\n~8 min (water)", "#8E24AA", 4),
]

y_center = 0.5
box_height = 0.7
box_width = 2.2
gap = 0.5

for label, time_text, color, idx in stages:
    x = idx * (box_width + gap)

    # Box
    rect = mpatches.FancyBboxPatch((x, y_center - box_height / 2), box_width, box_height,
                                     boxstyle="round,pad=0.15", facecolor=color, alpha=0.85,
                                     edgecolor="white", linewidth=2)
    ax.add_patch(rect)

    # Stage name (inside box)
    ax.text(x + box_width / 2, y_center + 0.05, label,
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    # Time below (larger text)
    ax.text(x + box_width / 2, y_center - 0.65, time_text,
            ha="center", va="top", fontsize=11, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Arrow to next
    if idx < len(stages) - 1:
        arrow_start = x + box_width + 0.05
        arrow_end = x + box_width + gap - 0.05
        ax.annotate("", xy=(arrow_end, y_center),
                    xytext=(arrow_start, y_center),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=2.5))

ax.set_xlim(-0.4, len(stages) * (box_width + gap) - gap + 0.4)
ax.set_ylim(-1.0, 1.4)
ax.axis("off")
ax.set_title("DeePMD pipeline: from DFT to production MD", fontweight="bold", fontsize=16, pad=20)

fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
