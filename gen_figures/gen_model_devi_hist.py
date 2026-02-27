"""
Generate model deviation histogram with trust_lo/trust_hi for Ch 7/10.
Uses synthetic data mimicking real distributions at different convergence stages.
Output: content/assets/plots/model_devi_hist.png
"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

trust_lo = 0.05
trust_hi = 0.15

# Iteration 0: broad distribution
iter0 = np.concatenate([
    np.random.exponential(0.06, 300),
    np.random.exponential(0.12, 200),
    np.random.exponential(0.25, 100),
])

# Iteration 2: tightening
iter2 = np.concatenate([
    np.random.exponential(0.03, 500),
    np.random.exponential(0.08, 150),
    np.random.exponential(0.20, 50),
])

# Iteration 4: converged
iter4 = np.concatenate([
    np.random.exponential(0.015, 700),
    np.random.exponential(0.05, 80),
    np.random.exponential(0.15, 20),
])

for ax, data, title in zip(axes, [iter0, iter2, iter4],
                            ["Iteration 0", "Iteration 2", "Iteration 4"]):
    data = data[data > 0]
    ax.hist(data, bins=80, range=(0, 0.4), color="steelblue",
            edgecolor="black", alpha=0.7, density=True)
    ax.axvline(trust_lo, color="green", linestyle="--", lw=2, label=f"trust_lo={trust_lo}")
    ax.axvline(trust_hi, color="red", linestyle="--", lw=2, label=f"trust_hi={trust_hi}")
    ax.axvspan(0, trust_lo, alpha=0.08, color="green")
    ax.axvspan(trust_lo, trust_hi, alpha=0.08, color="orange")
    ax.axvspan(trust_hi, 0.4, alpha=0.08, color="red")
    ax.set_xlabel("max_devi_f (eV/Å)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 0.4)

    # Compute fractions
    acc = np.mean(data < trust_lo) * 100
    cand = np.mean((data >= trust_lo) & (data < trust_hi)) * 100
    fail = np.mean(data >= trust_hi) * 100
    ax.text(0.95, 0.95, f"Accurate: {acc:.0f}%\nCandidate: {cand:.0f}%\nFailed: {fail:.0f}%",
           transform=ax.transAxes, fontsize=10, verticalalignment="top",
           horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

axes[0].set_ylabel("Density", fontsize=10)
axes[0].legend(fontsize=9, loc="upper center")

plt.suptitle("Force Deviation Distribution Over Iterations", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("content/assets/plots/model_devi_hist.png", dpi=200, bbox_inches="tight")
print("Saved: content/assets/plots/model_devi_hist.png")
