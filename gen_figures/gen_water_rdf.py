#!/usr/bin/env python3
"""Water RDF: O-O, O-H, H-H partial RDFs from NVT 300K."""
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os

traj_file = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd/03_lammps/nvt_300K/dump.lammpstrj"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "water_rdf.png")

# Water type_map: O=type1, H=type2 in LAMMPS (1-indexed)
# In ASE from LAMMPS dump, types are stored as numbers


def compute_partial_rdf(atoms_list, type_a, type_b, rmax=6.0, nbins=150):
    """Compute partial RDF between type_a and type_b atoms."""
    dr = rmax / nbins
    hist = np.zeros(nbins)

    for atoms in atoms_list:
        numbers = atoms.get_atomic_numbers()
        idx_a = np.where(numbers == type_a)[0]
        idx_b = np.where(numbers == type_b)[0]
        n_a = len(idx_a)
        n_b = len(idx_b)
        vol = atoms.get_volume()

        for i in idx_a:
            targets = idx_b if type_a != type_b else idx_b[idx_b > i]
            for j in targets:
                d = atoms.get_distance(i, j, mic=True)
                if d < rmax:
                    bin_idx = int(d / dr)
                    if bin_idx < nbins:
                        hist[bin_idx] += 1

        if type_a == type_b:
            hist *= 2  # symmetry

    n_frames = len(atoms_list)
    n_a_avg = np.mean([np.sum(atoms.get_atomic_numbers() == type_a) for atoms in atoms_list])
    n_b_avg = np.mean([np.sum(atoms.get_atomic_numbers() == type_b) for atoms in atoms_list])
    vol_avg = np.mean([atoms.get_volume() for atoms in atoms_list])
    rho_b = n_b_avg / vol_avg

    r = (np.arange(nbins) + 0.5) * dr
    shell_vol = 4 * np.pi * r**2 * dr
    rdf = hist / (n_frames * n_a_avg * rho_b * shell_vol)

    return r, rdf


print("Reading trajectory (this may take a moment)...")
all_frames = read(traj_file, index=":", format="lammps-dump-text")

# LAMMPS dump: atoms have atomic numbers from masses (O=8, H=1)
# But LAMMPS dump-text may assign Z from type: type1=O(8), type2=H(1)
# Check what we actually get
z_set = set(all_frames[0].get_atomic_numbers())
print(f"Atomic numbers in trajectory: {z_set}")

# If types are just 1,2 (not real Z), map them
if z_set == {1, 2}:
    # type 1 = O, type 2 = H (from LAMMPS input mass ordering)
    type_O, type_H = 1, 2
elif 8 in z_set:
    type_O, type_H = 8, 1
else:
    raise ValueError(f"Unexpected atomic numbers: {z_set}")

n_equil = len(all_frames) // 5
frames = all_frames[n_equil::10]
if len(frames) > 30:
    frames = frames[:30]

print(f"Computing RDFs from {len(frames)} frames...")

fig, ax = plt.subplots(figsize=(8, 5))

for label, ta, tb, color in [("O-O", type_O, type_O, "#2196F3"),
                               ("O-H", type_O, type_H, "#FF5722"),
                               ("H-H", type_H, type_H, "#4CAF50")]:
    r, g = compute_partial_rdf(frames, ta, tb, rmax=6.0, nbins=150)
    ax.plot(r, g, label=label, color=color, linewidth=1.5)

ax.set_xlabel("r (\u00c5)")
ax.set_ylabel("g(r)")
ax.set_title("Water partial RDFs from NVT 300 K (DeePMD)", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_xlim(0.5, 6.0)

fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
