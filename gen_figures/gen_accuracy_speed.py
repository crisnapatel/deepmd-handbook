"""
Generate accuracy vs speed bubble chart for Ch 1.
Shows DFT, Classical FF, and MLIP positions on the tradeoff.
Output: content/assets/plots/accuracy_speed.png
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))

# Data: (speed_log10, accuracy, label, size, color)
methods = [
    (1, 9.5, "DFT\n(QE/VASP)", 600, "#2196F3"),
    (4, 8.5, "MLIP\n(DeePMD)", 500, "#4CAF50"),
    (6, 5.0, "Classical FF\n(AIREBO, ReaxFF)", 500, "#FF9800"),
    (7, 3.0, "Empirical\n(LJ, Morse)", 400, "#F44336"),
]

for speed, acc, label, size, color in methods:
    ax.scatter(speed, acc, s=size, c=color, alpha=0.7, edgecolors="black",
              linewidth=1.5, zorder=3)
    ax.annotate(label, (speed, acc), textcoords="offset points",
               xytext=(0, -35), ha="center", fontsize=9, fontweight="bold")

# Arrow showing the sweet spot
ax.annotate("", xy=(4, 8.5), xytext=(1.5, 7),
           arrowprops=dict(arrowstyle="->", color="green", lw=2))
ax.text(2.2, 6.5, "DFT accuracy\nat FF speed", fontsize=10, color="green",
       style="italic")

ax.set_xlabel("Computational Speed (log scale)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("The Accuracy-Speed Tradeoff", fontsize=14)
ax.set_xlim(0, 8)
ax.set_ylim(1, 11)
ax.set_xticks([1, 4, 6, 7])
ax.set_xticklabels(["100 atoms\n10 ps", "10k atoms\n10 ns", "1M atoms\n100 ns",
                     "10M atoms\n1 μs"])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("content/assets/plots/accuracy_speed.png", dpi=200, bbox_inches="tight")
print("Saved: content/assets/plots/accuracy_speed.png")
