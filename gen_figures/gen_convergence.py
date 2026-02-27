"""
Generate convergence plot: accurate fraction vs iteration for Ch 10.
Output: content/assets/plots/convergence.png
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))

iterations = np.arange(0, 8)
acc_frac = np.array([0.30, 0.55, 0.75, 0.88, 0.94, 0.97, 0.985, 0.993])

ax.plot(iterations, acc_frac, "o-", color="steelblue", linewidth=2.5,
       markersize=10, markeredgecolor="black", markeredgewidth=1.5, zorder=3)

ax.axhline(0.98, color="green", linestyle="--", linewidth=1.5,
          label="fp_accurate_threshold (0.98)", alpha=0.8)
ax.axhline(0.90, color="orange", linestyle="--", linewidth=1.5,
          label="fp_accurate_soft_threshold (0.90)", alpha=0.8)

ax.fill_between(iterations, acc_frac, 0, alpha=0.1, color="steelblue")

ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Accurate Fraction", fontsize=12)
ax.set_title("dpgen Convergence", fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_xlim(-0.3, 7.3)
ax.set_xticks(iterations)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("content/assets/plots/convergence.png", dpi=200, bbox_inches="tight")
print("Saved: content/assets/plots/convergence.png")
