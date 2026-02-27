#!/usr/bin/env python3
"""Virial demo: Ar virial parity plot only (text moved to chapter)."""
import numpy as np
import matplotlib.pyplot as plt
import os

test_dir = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/03_test"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "virial_demo.png")

fig, ax = plt.subplots(figsize=(7, 6))

colors = {"FCC": "#2196F3", "Liquid": "#FF5722"}
for label, prefix in [("FCC", "test_fcc"), ("Liquid", "test_liquid")]:
    v_data = np.loadtxt(os.path.join(test_dir, f"{prefix}.v_peratom.out"))
    dft_v = v_data[:, :9].ravel()
    dp_v = v_data[:, 9:].ravel()
    ax.scatter(dft_v, dp_v, s=8, alpha=0.4, color=colors[label], label=label, edgecolors="none")

all_v = np.concatenate([line.get_offsets()[:, 0] for line in ax.collections])
lo, hi = all_v.min(), all_v.max()
margin = 0.05 * (hi - lo)
ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.5, linewidth=0.8)

# Compute RMSE
all_dft = []
all_dp = []
for prefix in ["test_fcc", "test_liquid"]:
    v_data = np.loadtxt(os.path.join(test_dir, f"{prefix}.v_peratom.out"))
    all_dft.append(v_data[:, :9].ravel())
    all_dp.append(v_data[:, 9:].ravel())
all_dft = np.concatenate(all_dft)
all_dp = np.concatenate(all_dp)
v_rmse = np.sqrt(np.mean((all_dft - all_dp) ** 2)) * 1000

ax.annotate(f"Virial RMSE = {v_rmse:.2f} meV/atom",
            xy=(0.05, 0.93), xycoords="axes fraction", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.set_xlabel("DFT virial (eV/atom)", fontsize=12)
ax.set_ylabel("DP virial (eV/atom)", fontsize=12)
ax.set_title("Ar virial parity: trained WITH virial data", fontsize=14, fontweight="bold")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(fontsize=11, markerscale=3)
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
