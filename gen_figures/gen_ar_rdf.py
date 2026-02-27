#!/usr/bin/env python3
"""Ar RDF: Ar-Ar RDF comparing sharp FCC peaks vs broad liquid."""
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os

lammps_root = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/04_lammps"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "ar_rdf.png")


def compute_rdf(atoms_list, rmax=8.0, nbins=200):
    """Compute RDF from a list of ASE Atoms objects."""
    dr = rmax / nbins
    hist = np.zeros(nbins)

    for atoms in atoms_list:
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        n = len(atoms)
        vol = atoms.get_volume()

        for i in range(n):
            for j in range(i + 1, n):
                d = atoms.get_distance(i, j, mic=True)
                if d < rmax:
                    bin_idx = int(d / dr)
                    if bin_idx < nbins:
                        hist[bin_idx] += 2  # count i-j and j-i

    # Normalize
    n_frames = len(atoms_list)
    n_atoms = len(atoms_list[0])
    rho = n_atoms / atoms_list[0].get_volume()

    r = (np.arange(nbins) + 0.5) * dr
    shell_vol = 4 * np.pi * r**2 * dr
    rdf = hist / (n_frames * n_atoms * rho * shell_vol)

    return r, rdf


fig, ax = plt.subplots(figsize=(8, 5))

for label, subdir, color in [("FCC (50 K)", "nvt_solid", "#2196F3"),
                               ("Liquid (150 K)", "nvt_liquid", "#FF5722")]:
    traj_file = os.path.join(lammps_root, subdir, "dump.lammpstrj")
    # Read every 10th frame, skip first 20% for equilibration
    all_frames = read(traj_file, index=":", format="lammps-dump-text")
    n_equil = len(all_frames) // 5
    frames = all_frames[n_equil::10]

    if len(frames) > 30:
        frames = frames[:30]  # cap for speed

    r, g = compute_rdf(frames, rmax=8.0, nbins=200)
    ax.plot(r, g, label=label, color=color, linewidth=1.5)

ax.set_xlabel("r (\u00c5)")
ax.set_ylabel("g(r)")
ax.set_title("Ar-Ar radial distribution function", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_xlim(2, 8)

fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
