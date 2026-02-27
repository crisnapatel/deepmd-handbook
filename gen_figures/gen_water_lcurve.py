#!/usr/bin/env python3
"""Water learning curve: 3-panel energy/force RMSE + lr vs steps."""
import numpy as np
import matplotlib.pyplot as plt
import os

lcurve_file = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd/01_train/lcurve.out"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "water_lcurve.png")

# Columns: step rmse_val rmse_trn rmse_e_val rmse_e_trn rmse_f_val rmse_f_trn lr
data = np.loadtxt(lcurve_file)
step = data[:, 0]
e_val, e_trn = data[:, 3], data[:, 4]
f_val, f_trn = data[:, 5], data[:, 6]
lr = data[:, 7]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Energy RMSE
ax = axes[0]
ax.semilogy(step / 1000, e_trn, alpha=0.6, label="Train", color="#2196F3", linewidth=0.8)
ax.semilogy(step / 1000, e_val, alpha=0.6, label="Validation", color="#FF5722", linewidth=0.8)
ax.set_xlabel("Training step (\u00d71000)")
ax.set_ylabel("Energy RMSE (eV)")
ax.set_title("Energy")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Force RMSE
ax = axes[1]
ax.semilogy(step / 1000, f_trn, alpha=0.6, label="Train", color="#2196F3", linewidth=0.8)
ax.semilogy(step / 1000, f_val, alpha=0.6, label="Validation", color="#FF5722", linewidth=0.8)
ax.set_xlabel("Training step (\u00d71000)")
ax.set_ylabel("Force RMSE (eV/\u00c5)")
ax.set_title("Force")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Learning rate
ax = axes[2]
ax.semilogy(step / 1000, lr, color="#4CAF50", linewidth=1)
ax.set_xlabel("Training step (\u00d71000)")
ax.set_ylabel("Learning rate")
ax.set_title("Learning rate schedule")
ax.grid(True, alpha=0.3)

fig.suptitle("Water model: 500k steps, se_e2_a, rcut=6.0 \u00c5, no virial", fontweight="bold")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
