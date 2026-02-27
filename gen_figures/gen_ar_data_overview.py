#!/usr/bin/env python3
"""Ar data overview: energy + force magnitude distributions, FCC vs liquid."""
import numpy as np
import matplotlib.pyplot as plt
import os

data_root = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/01_data"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "ar_data_overview.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = {"FCC (50 K)": "#2196F3", "Liquid (150 K)": "#FF5722"}

for label, phase in [("FCC (50 K)", "ar_fcc"), ("Liquid (150 K)", "ar_liquid")]:
    # Combine training + validation
    energies = []
    forces = []
    for split in ["training", "validation"]:
        base = os.path.join(data_root, split, phase, "set.000")
        e = np.load(os.path.join(base, "energy.npy"))
        f = np.load(os.path.join(base, "force.npy"))
        energies.append(e)
        forces.append(f)

    energies = np.concatenate(energies)
    forces = np.concatenate(forces)

    natoms = forces.shape[1]
    e_per_atom = energies / natoms

    # Force magnitudes per atom
    f_mag = np.linalg.norm(forces.reshape(-1, 3), axis=1)

    c = colors[label]
    axes[0].hist(e_per_atom, bins=30, alpha=0.6, label=label, color=c, edgecolor="white", linewidth=0.5)
    axes[1].hist(f_mag, bins=50, alpha=0.6, label=label, color=c, edgecolor="white", linewidth=0.5, density=True)

axes[0].set_xlabel("Energy per atom (eV)")
axes[0].set_ylabel("Count")
axes[0].set_title("Energy distribution")
axes[0].legend()

axes[1].set_xlabel("Force magnitude (eV/\u00c5)")
axes[1].set_ylabel("Probability density")
axes[1].set_title("Force magnitude distribution")
axes[1].legend()

fig.suptitle("Ar training data: 100 FCC + 100 liquid frames, 32 atoms", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
