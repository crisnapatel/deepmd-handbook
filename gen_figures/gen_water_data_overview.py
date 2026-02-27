#!/usr/bin/env python3
"""Water data overview: energy + force distributions across data_0/1/2/3."""
import numpy as np
import matplotlib.pyplot as plt
import os

data_root = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd/00_data"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "water_data_overview.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

datasets = {
    "data_0 (train)": ("training", "data_0"),
    "data_1 (train)": ("training", "data_1"),
    "data_2 (train)": ("training", "data_2"),
    "data_3 (val)": ("validation", "data_3"),
}
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, 4))

for idx, (label, (split, dname)) in enumerate(datasets.items()):
    base = os.path.join(data_root, split, dname)
    # Collect from all set.* directories
    energies = []
    forces = []
    for sdir in sorted(os.listdir(base)):
        if sdir.startswith("set."):
            spath = os.path.join(base, sdir)
            e_file = os.path.join(spath, "energy.npy")
            f_file = os.path.join(spath, "force.npy")
            if os.path.exists(e_file):
                energies.append(np.load(e_file))
            if os.path.exists(f_file):
                forces.append(np.load(f_file))

    if not energies:
        continue

    energies = np.concatenate(energies)
    forces = np.concatenate(forces)

    natoms = forces.shape[1]
    e_per_atom = energies / natoms

    f_mag = np.linalg.norm(forces.reshape(-1, 3), axis=1)

    c = cmap[idx]
    axes[0].hist(e_per_atom, bins=25, alpha=0.5, label=label, color=c, edgecolor="white", linewidth=0.5)
    axes[1].hist(f_mag, bins=50, alpha=0.5, label=label, color=c, edgecolor="white", linewidth=0.5, density=True)

axes[0].set_xlabel("Energy per atom (eV)")
axes[0].set_ylabel("Count")
axes[0].set_title("Energy distribution")
axes[0].legend(fontsize=9)

axes[1].set_xlabel("Force magnitude (eV/\u00c5)")
axes[1].set_ylabel("Probability density")
axes[1].set_title("Force magnitude distribution")
axes[1].legend(fontsize=9)

fig.suptitle("Water training data: 320 train + 80 val frames, 192 atoms (64 H\u2082O)", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
