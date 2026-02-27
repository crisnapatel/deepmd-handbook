#!/usr/bin/env python3
"""Water structure: 192-atom water box with O-H bonds, atoms wrapped into cell."""
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
import os

data_root = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd/00_data"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "water_structure.png")

# Read type_map to get element ordering
type_map_file = os.path.join(data_root, "training", "data_0", "type_map.raw")
with open(type_map_file) as f:
    type_map = f.read().split()

# Read type.raw
type_raw_file = os.path.join(data_root, "training", "data_0", "type.raw")
with open(type_raw_file) as f:
    types = [int(x) for x in f.read().split()]

base = os.path.join(data_root, "training", "data_0", "set.000")
coords = np.load(os.path.join(base, "coord.npy"))
boxes = np.load(os.path.join(base, "box.npy"))

# First frame
pos = coords[0].reshape(-1, 3)
box = boxes[0].reshape(3, 3)

# Build chemical symbols from type_map + types
symbols = [type_map[t] for t in types]
atoms = Atoms(symbols, positions=pos, cell=box, pbc=True)

# Wrap all atoms into the unit cell so nothing sticks out
atoms.wrap()
positions = atoms.get_positions()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Get cell for PBC check
cell = atoms.get_cell()

# Draw O-H bonds using minimum image convention
# For each O, find H atoms within 1.2 A using MIC
o_indices = [i for i, s in enumerate(symbols) if s == 'O']
h_indices = [i for i, s in enumerate(symbols) if s == 'H']

for oi in o_indices:
    for hi in h_indices:
        d = atoms.get_distance(oi, hi, mic=True)
        if d < 1.2:
            # Only draw if both atoms are close in unwrapped coordinates too
            # (skip bonds that cross the periodic boundary in the xy projection)
            p_o = positions[oi]
            p_h = positions[hi]
            dx = abs(p_o[0] - p_h[0])
            dy = abs(p_o[1] - p_h[1])
            # If the real-space distance in x or y is > half the cell, it crosses PBC
            if dx > cell[0, 0] / 2 or dy > cell[1, 1] / 2:
                continue
            ax.plot([p_o[0], p_h[0]], [p_o[1], p_h[1]],
                    color='#888888', linewidth=0.8, zorder=1)

# Draw atoms: O large red, H small white
for i, (sym, p) in enumerate(zip(symbols, positions)):
    if sym == 'O':
        ax.scatter(p[0], p[1], s=120, c='#E53935', edgecolors='#B71C1C',
                   linewidth=0.8, zorder=3)
    else:
        ax.scatter(p[0], p[1], s=30, c='white', edgecolors='#424242',
                   linewidth=0.5, zorder=2)

# Draw cell box
cell = atoms.get_cell()
corners = np.array([
    [0, 0],
    [cell[0, 0], cell[0, 1]],
    [cell[0, 0] + cell[1, 0], cell[0, 1] + cell[1, 1]],
    [cell[1, 0], cell[1, 1]],
    [0, 0]
])
ax.plot(corners[:, 0], corners[:, 1], 'k-', linewidth=1.0, alpha=0.5)

# Set axis limits to the cell boundaries with small padding
x_coords = corners[:-1, 0]
y_coords = corners[:-1, 1]
pad = 0.5
ax.set_xlim(min(x_coords) - pad, max(x_coords) + pad)
ax.set_ylim(min(y_coords) - pad, max(y_coords) + pad)

ax.set_xlabel("x (\u00c5)", fontsize=12)
ax.set_ylabel("y (\u00c5)", fontsize=12)
ax.set_title(f"Water: {len(atoms)} atoms ({symbols.count('O')} O + {symbols.count('H')} H)",
             fontsize=14, fontweight="bold")
ax.set_aspect("equal")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935',
           markeredgecolor='#B71C1C', markersize=12, label='Oxygen'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='#424242', markersize=7, label='Hydrogen'),
]
ax.legend(handles=legend_elements, fontsize=11, loc='upper right')

fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
