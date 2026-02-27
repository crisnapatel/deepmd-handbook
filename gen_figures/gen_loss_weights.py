"""
Generate loss weight evolution during training for Ch 2.
Shows how start_pref_f/e transition to limit_pref_f/e.
Output: content/assets/plots/loss_weights.png
"""
import matplotlib.pyplot as plt
import numpy as np

steps = np.linspace(0, 1e6, 1000)
numb_steps = 1e6

# DeePMD interpolation: pref(t) = start * (limit/start)^(t/numb_steps)
pref_f = 1000 * (1.0 / 1000) ** (steps / numb_steps)  # 1000 -> 1
pref_e = 0.02 * (2.0 / 0.02) ** (steps / numb_steps)   # 0.02 -> 2

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(steps / 1000, pref_f, label="Force prefactor ($p_f$)",
           color="#2196F3", linewidth=2.5)
ax.semilogy(steps / 1000, pref_e, label="Energy prefactor ($p_e$)",
           color="#FF5722", linewidth=2.5)

# Annotations — positioned away from lines and legend
ax.annotate("Forces dominate\n(learning the PES shape)",
           xy=(50, 500), fontsize=9, color="#2196F3",
           bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.annotate("Energy catches up\n(calibrating absolute scale)",
           xy=(350, 0.04), fontsize=9, color="#FF5722",
           bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# Crossover point
cross_idx = np.argmin(np.abs(pref_f - pref_e))
ax.axvline(steps[cross_idx] / 1000, color="gray", linestyle=":", alpha=0.5)
ax.text(steps[cross_idx] / 1000 + 20, 30, "Crossover", fontsize=10, color="gray")

ax.set_xlabel("Training Step (×10³)", fontsize=12)
ax.set_ylabel("Loss Prefactor", fontsize=12)
ax.set_title("Loss Weight Evolution During Training", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 1000)

plt.tight_layout()
plt.savefig("content/assets/plots/loss_weights.png", dpi=200, bbox_inches="tight")
print("Saved: content/assets/plots/loss_weights.png")
