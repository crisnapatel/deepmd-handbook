#!/usr/bin/env python3
"""Ar parity plots: 3-panel energy/force/virial DFT vs DP (FCC+liquid combined)."""
import numpy as np
import matplotlib.pyplot as plt
import os

test_dir = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/03_test"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "ar_parity.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

colors = {"FCC": "#2196F3", "Liquid": "#FF5722"}

for label, prefix in [("FCC", "test_fcc"), ("Liquid", "test_liquid")]:
    c = colors[label]

    # Energy per atom
    e_data = np.loadtxt(os.path.join(test_dir, f"{prefix}.e_peratom.out"))
    dft_e, dp_e = e_data[:, 0], e_data[:, 1]

    # Forces (data_fx data_fy data_fz pred_fx pred_fy pred_fz)
    f_data = np.loadtxt(os.path.join(test_dir, f"{prefix}.f.out"))
    dft_f = f_data[:, :3].ravel()
    dp_f = f_data[:, 3:].ravel()

    # Virial per atom
    v_data = np.loadtxt(os.path.join(test_dir, f"{prefix}.v_peratom.out"))
    dft_v = v_data[:, :9].ravel()
    dp_v = v_data[:, 9:].ravel()

    # Energy parity
    axes[0].scatter(dft_e, dp_e, s=20, alpha=0.7, label=label, color=c, edgecolors="none")

    # Force parity (subsample for clarity)
    idx = np.random.RandomState(42).choice(len(dft_f), min(5000, len(dft_f)), replace=False)
    axes[1].scatter(dft_f[idx], dp_f[idx], s=3, alpha=0.3, label=label, color=c, edgecolors="none")

    # Virial parity
    axes[2].scatter(dft_v, dp_v, s=5, alpha=0.4, label=label, color=c, edgecolors="none")

for i, (ax, title, unit) in enumerate(zip(axes,
    ["Energy (per atom)", "Force components", "Virial (per atom)"],
    ["eV/atom", "eV/\u00c5", "eV/atom"])):

    # y=x line
    all_pts = np.concatenate([line.get_offsets()[:, 0] for line in ax.collections])
    lo, hi = all_pts.min(), all_pts.max()
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel(f"DFT ({unit})")
    ax.set_ylabel(f"DP ({unit})")
    ax.set_title(title)
    ax.legend(fontsize=9, markerscale=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

fig.suptitle("Ar model: DFT vs DeePMD parity (validation set)", fontweight="bold")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
