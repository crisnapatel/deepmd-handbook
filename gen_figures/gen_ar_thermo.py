#!/usr/bin/env python3
"""Ar thermo: T + Etotal vs time for NVT solid, shows stability."""
import numpy as np
import matplotlib.pyplot as plt
import os

log_file = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/04_lammps/nvt_solid/log.lammps"
out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "ar_thermo.png")

# Parse LAMMPS log: find thermo data between "Step ..." and "Loop ..."
steps, temps, pe, ke, etot, press = [], [], [], [], [], []

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
        if len(parts) >= 5:
            try:
                steps.append(float(parts[0]))
                temps.append(float(parts[1]))
                pe.append(float(parts[2]))
                ke.append(float(parts[3]))
                etot.append(float(parts[4]))
            except ValueError:
                continue

steps = np.array(steps)
temps = np.array(temps)
etot = np.array(etot)

# timestep is 0.001 ps = 1 fs
time_ps = steps * 0.001

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

axes[0].plot(time_ps, temps, color="#FF5722", linewidth=0.8, alpha=0.8)
axes[0].axhline(50, color="gray", linestyle="--", alpha=0.5, label="Target: 50 K")
axes[0].set_ylabel("Temperature (K)")
axes[0].set_title("NVT thermostat stability")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.2)

axes[1].plot(time_ps, etot, color="#2196F3", linewidth=0.8, alpha=0.8)
axes[1].set_xlabel("Time (ps)")
axes[1].set_ylabel("Total energy (eV)")
axes[1].set_title("Energy conservation")
axes[1].grid(True, alpha=0.2)

fig.suptitle("Ar FCC: 10 ps NVT at 50 K with DeePMD potential", fontweight="bold")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
