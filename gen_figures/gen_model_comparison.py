#!/usr/bin/env python3
"""Model comparison: bar chart of Ar vs Water accuracy metrics side by side."""
import numpy as np
import matplotlib.pyplot as plt
import os

out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "model_comparison.png")

# Compute actual RMSEs from test data
test_ar = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/03_test"
test_water = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd"

# Ar: combine FCC + liquid
ar_e_fcc = np.loadtxt(os.path.join(test_ar, "test_fcc.e_peratom.out"))
ar_e_liq = np.loadtxt(os.path.join(test_ar, "test_liquid.e_peratom.out"))
ar_f_fcc = np.loadtxt(os.path.join(test_ar, "test_fcc.f.out"))
ar_f_liq = np.loadtxt(os.path.join(test_ar, "test_liquid.f.out"))

ar_e_rmse = np.sqrt(np.mean(np.concatenate([
    (ar_e_fcc[:, 0] - ar_e_fcc[:, 1])**2,
    (ar_e_liq[:, 0] - ar_e_liq[:, 1])**2
]))) * 1000  # meV/atom

ar_f_fcc_err = ar_f_fcc[:, :3] - ar_f_fcc[:, 3:]
ar_f_liq_err = ar_f_liq[:, :3] - ar_f_liq[:, 3:]
ar_f_rmse = np.sqrt(np.mean(np.concatenate([ar_f_fcc_err.ravel()**2, ar_f_liq_err.ravel()**2]))) * 1000  # meV/A

# Water
w_e = np.loadtxt(os.path.join(test_water, "02_test.e_peratom.out"))
w_f = np.loadtxt(os.path.join(test_water, "02_test.f.out"))

w_e_rmse = np.sqrt(np.mean((w_e[:, 0] - w_e[:, 1])**2)) * 1000
w_f_err = w_f[:, :3] - w_f[:, 3:]
w_f_rmse = np.sqrt(np.mean(w_f_err.ravel()**2)) * 1000

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Energy RMSE
x = np.arange(2)
bars = axes[0].bar(x, [ar_e_rmse, w_e_rmse], color=["#2196F3", "#FF5722"], width=0.5, edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Ar", "Water"])
axes[0].set_ylabel("Energy RMSE (meV/atom)")
axes[0].set_title("Energy accuracy")
for bar, val in zip(bars, [ar_e_rmse, w_e_rmse]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].grid(True, alpha=0.2, axis="y")

# Force RMSE
bars = axes[1].bar(x, [ar_f_rmse, w_f_rmse], color=["#2196F3", "#FF5722"], width=0.5, edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Ar", "Water"])
axes[1].set_ylabel("Force RMSE (meV/\u00c5)")
axes[1].set_title("Force accuracy")
for bar, val in zip(bars, [ar_f_rmse, w_f_rmse]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].grid(True, alpha=0.2, axis="y")

fig.suptitle("Model accuracy comparison: Ar (single element) vs Water (multi-element)", fontweight="bold")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
