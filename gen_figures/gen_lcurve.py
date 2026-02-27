"""
Generate synthetic learning curve plot for Ch 4.
Mimics a typical lcurve.out from dp train.
Output: content/assets/plots/lcurve.png
"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

steps = np.arange(0, 400001, 1000)
lr = 0.001 * (1e-7 / 0.001) ** (steps / 400000)

# Synthetic loss curves (exponential decay + noise)
e_train = 5.0 * np.exp(-steps / 30000) + 0.0005 + np.random.normal(0, 0.0002, len(steps))
e_val = 5.5 * np.exp(-steps / 28000) + 0.0007 + np.random.normal(0, 0.0003, len(steps))
f_train = 20.0 * np.exp(-steps / 20000) + 0.03 + np.random.normal(0, 0.002, len(steps))
f_val = 22.0 * np.exp(-steps / 18000) + 0.04 + np.random.normal(0, 0.003, len(steps))

# Clip negatives
e_train = np.clip(e_train, 1e-5, None)
e_val = np.clip(e_val, 1e-5, None)
f_train = np.clip(f_train, 1e-3, None)
f_val = np.clip(f_val, 1e-3, None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.semilogy(steps, e_train, label="Train", color="steelblue", alpha=0.8)
ax1.semilogy(steps, e_val, label="Validation", color="coral", alpha=0.8)
ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Energy RMSE (eV)", fontsize=11)
ax1.set_title("Energy Loss", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

ax2.semilogy(steps, f_train, label="Train", color="steelblue", alpha=0.8)
ax2.semilogy(steps, f_val, label="Validation", color="coral", alpha=0.8)
ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Force RMSE (eV/Å)", fontsize=11)
ax2.set_title("Force Loss", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

plt.suptitle("Learning Curve (lcurve.out)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("content/assets/plots/lcurve.png", dpi=200, bbox_inches="tight")
print("Saved: content/assets/plots/lcurve.png")
