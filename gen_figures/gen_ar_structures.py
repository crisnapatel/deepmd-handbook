#!/usr/bin/env python3
"""Ar structures: side-by-side FCC vs liquid snapshots using ASE plot_atoms."""
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.visualize.plot import plot_atoms
import os

data_root = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/01_data"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "ar_structures.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, (label, phase) in zip(axes, [("FCC (50 K)", "ar_fcc"), ("Liquid (150 K)", "ar_liquid")]):
    base = os.path.join(data_root, "training", phase, "set.000")
    coords = np.load(os.path.join(base, "coord.npy"))  # (nframes, natoms*3) or (nframes, natoms, 3)
    boxes = np.load(os.path.join(base, "box.npy"))

    # Take first frame
    pos = coords[0].reshape(-1, 3)
    box = boxes[0].reshape(3, 3)

    atoms = Atoms("Ar" * len(pos), positions=pos, cell=box, pbc=True)

    plot_atoms(atoms, ax, radii=1.5, rotation="10x,10y,0z")
    ax.set_title(label, fontsize=14, fontweight="bold")
    ax.set_xlabel("\u00c5")
    ax.set_ylabel("\u00c5")

fig.suptitle("Ar: 32-atom supercell structures", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
