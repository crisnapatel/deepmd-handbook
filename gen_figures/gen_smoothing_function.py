"""
Generate smoothing function plot for Ch 2.
Shows how atom contributions taper between rcut_smth and rcut.
Replaces the screenshot from lecture slides.
Output: content/assets/plots/smoothing_function.png
"""
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Parameters
rcut = 6.0
rcut_smth = 2.0

r = np.linspace(0, 8, 500)

# Smoothing function: s(r)
# s(r) = 1                                          if r < rcut_smth
# s(r) = 0.5 * cos(pi * (r - rcut_smth) / (rcut - rcut_smth)) + 0.5   if rcut_smth <= r < rcut
# s(r) = 0                                          if r >= rcut
s = np.where(r < rcut_smth, 1.0,
     np.where(r < rcut,
              0.5 * np.cos(np.pi * (r - rcut_smth) / (rcut - rcut_smth)) + 0.5,
              0.0))

# Left panel: smoothing function curve
ax1.plot(r, s, color='#1565C0', linewidth=2.5, zorder=3)
ax1.axvline(rcut_smth, color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'rcut_smth = {rcut_smth} Å')
ax1.axvline(rcut, color='#F44336', linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'rcut = {rcut} Å')

# Shaded regions
ax1.axvspan(0, rcut_smth, alpha=0.08, color='#4CAF50')
ax1.axvspan(rcut_smth, rcut, alpha=0.08, color='#FF9800')
ax1.axvspan(rcut, 8, alpha=0.05, color='#F44336')

# Labels for regions
ax1.text(1.0, 0.5, 'Full\ncontribution', ha='center', fontsize=9,
         color='#2E7D32', fontweight='bold')
ax1.text(4.0, 0.5, 'Smooth\ntaper', ha='center', fontsize=9,
         color='#E65100', fontweight='bold')
ax1.text(7.0, 0.5, 'Invisible', ha='center', fontsize=9,
         color='#C62828', fontweight='bold')

ax1.set_xlabel('Distance r (Å)', fontsize=11)
ax1.set_ylabel('s(r)', fontsize=11)
ax1.set_title('Smoothing Function', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='center right')
ax1.set_xlim(0, 8)
ax1.set_ylim(-0.05, 1.15)
ax1.grid(True, alpha=0.15)

# Right panel: 2D schematic of cutoff sphere
theta = np.linspace(0, 2 * np.pi, 100)

# Draw cutoff circles
ax2.plot(rcut * np.cos(theta), rcut * np.sin(theta),
         color='#F44336', linewidth=2, linestyle='--', label=f'rcut = {rcut} Å')
ax2.plot(rcut_smth * np.cos(theta), rcut_smth * np.sin(theta),
         color='#4CAF50', linewidth=2, linestyle='--', label=f'rcut_smth = {rcut_smth} Å')

# Shaded regions
circle_r = plt.Circle((0, 0), rcut_smth, color='#4CAF50', alpha=0.1)
ring = plt.matplotlib.patches.Annulus((0, 0), rcut, rcut_smth, color='#FF9800', alpha=0.1)
ax2.add_patch(circle_r)
ax2.add_patch(ring)

# Central atom
ax2.plot(0, 0, 'o', color='#1565C0', markersize=14, markeredgecolor='black',
         markeredgewidth=1.5, zorder=5)
ax2.text(0, -0.6, 'atom i', ha='center', fontsize=9, color='#1565C0', fontweight='bold')

# Neighbor atoms at various distances
np.random.seed(42)
for dist_range, color, alpha, label in [
    ((0.5, 1.8), '#4CAF50', 1.0, 'full weight'),
    ((2.5, 5.5), '#FF9800', 0.6, 'tapered'),
    ((6.5, 7.5), '#BDBDBD', 0.3, 'invisible'),
]:
    n_atoms = 5
    angles = np.random.uniform(0, 2 * np.pi, n_atoms)
    dists = np.random.uniform(dist_range[0], dist_range[1], n_atoms)
    xs = dists * np.cos(angles)
    ys = dists * np.sin(angles)
    ax2.scatter(xs, ys, c=color, s=60, edgecolors='black', linewidth=0.8,
                alpha=alpha, zorder=4)

ax2.set_xlim(-8.5, 8.5)
ax2.set_ylim(-8.5, 8.5)
ax2.set_aspect('equal')
ax2.set_xlabel('x (Å)', fontsize=11)
ax2.set_ylabel('y (Å)', fontsize=11)
ax2.set_title('Cutoff Sphere (Top View)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('content/assets/plots/smoothing_function.png', dpi=200, bbox_inches='tight')
print('Saved: content/assets/plots/smoothing_function.png')
