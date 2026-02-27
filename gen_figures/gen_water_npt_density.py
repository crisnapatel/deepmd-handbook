#!/usr/bin/env python3
"""Water NPT density vs time showing drift. Cite Gillan 2016, Jonchiere 2012."""
import numpy as np
import matplotlib.pyplot as plt
import os

log_file = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd/03_lammps/npt_300K/log.lammps"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "water_npt_density.png")

# Parse LAMMPS log: columns are Step Temp PotEng KinEng TotEng Press Vol Density
steps, density, vol = [], [], []

with open(log_file) as f:
    lines = f.readlines()

in_thermo = False
for line in lines:
    if line.strip().startswith("Step"):
        in_thermo = True
        continue
    if line.strip().startswith("Loop"):
        in_thermo = False
        continue
    if in_thermo:
        parts = line.split()
        if len(parts) >= 8:
            try:
                steps.append(float(parts[0]))
                vol.append(float(parts[6]))
                density.append(float(parts[7]))
            except ValueError:
                continue

steps = np.array(steps)
density = np.array(density)

# timestep is 0.0005 ps = 0.5 fs
time_ps = steps * 0.0005

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(time_ps, density, color="#2196F3", linewidth=0.8, alpha=0.8)

# Reference lines
ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Experiment (1.0 g/cm\u00b3)")
ax.axhline(0.79, color="#FF5722", linestyle=":", alpha=0.7, label="PBE-DFT (0.79, Jonchiere 2012)")
ax.axhline(0.77, color="#4CAF50", linestyle=":", alpha=0.7, label="BLYP-DFT (0.77, Jonchiere 2012)")

ax.set_xlabel("Time (ps)")
ax.set_ylabel("Density (g/cm\u00b3)")
ax.set_title("Water NPT 300 K: density evolution", fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.2)

# Annotate
avg_density = np.mean(density[len(density)//2:])
ax.annotate(f"Avg (last half): {avg_density:.3f} g/cm\u00b3",
            xy=(0.05, 0.05), xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
