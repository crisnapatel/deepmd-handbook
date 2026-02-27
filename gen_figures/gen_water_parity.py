#!/usr/bin/env python3
"""Water parity plots: 2-panel energy/force DFT vs DP."""
import numpy as np
import matplotlib.pyplot as plt
import os

test_dir = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "water_parity.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Energy per atom
e_data = np.loadtxt(os.path.join(test_dir, "02_test.e_peratom.out"))
dft_e, dp_e = e_data[:, 0], e_data[:, 1]

# Forces
f_data = np.loadtxt(os.path.join(test_dir, "02_test.f.out"))
dft_f = f_data[:, :3].ravel()
dp_f = f_data[:, 3:].ravel()

# Energy parity
axes[0].scatter(dft_e, dp_e, s=15, alpha=0.6, color="#2196F3", edgecolors="none")
lo, hi = dft_e.min(), dft_e.max()
margin = 0.05 * (hi - lo)
axes[0].plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.5, linewidth=0.8)
axes[0].set_xlabel("DFT energy (eV/atom)")
axes[0].set_ylabel("DP energy (eV/atom)")
axes[0].set_title("Energy (per atom)")
axes[0].set_aspect("equal", adjustable="datalim")
axes[0].grid(True, alpha=0.2)

# RMSE annotation
e_rmse = np.sqrt(np.mean((dft_e - dp_e) ** 2)) * 1000  # meV
axes[0].annotate(f"RMSE = {e_rmse:.2f} meV/atom", xy=(0.05, 0.92),
                 xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Force parity (subsample)
idx = np.random.RandomState(42).choice(len(dft_f), min(10000, len(dft_f)), replace=False)
axes[1].scatter(dft_f[idx], dp_f[idx], s=2, alpha=0.15, color="#FF5722", edgecolors="none")
lo, hi = dft_f[idx].min(), dft_f[idx].max()
margin = 0.05 * (hi - lo)
axes[1].plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.5, linewidth=0.8)
axes[1].set_xlabel("DFT force (eV/\u00c5)")
axes[1].set_ylabel("DP force (eV/\u00c5)")
axes[1].set_title("Force components")
axes[1].set_aspect("equal", adjustable="datalim")
axes[1].grid(True, alpha=0.2)

f_rmse = np.sqrt(np.mean((dft_f - dp_f) ** 2)) * 1000  # meV/A
axes[1].annotate(f"RMSE = {f_rmse:.1f} meV/\u00c5", xy=(0.05, 0.92),
                 xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

fig.suptitle("Water model: DFT vs DeePMD parity (80-frame validation set)", fontweight="bold")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
