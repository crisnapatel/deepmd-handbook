"""
Generate energy scale mismatch bar chart for Practical: Energy Scale.
Shows the incompatibility between isolated H2 and slab energies.
Output: content/assets/plots/energy_scale.png
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 5))

systems = ["Isolated H₂", "Graphene slab\n(72 C atoms)", "Graphene + 4 H₂\n(80 atoms)"]
energies = [-16.4, -278.3, -262.1]  # Approximate eV/atom
colors = ["#F44336", "#2196F3", "#4CAF50"]

bars = ax.bar(systems, energies, color=colors, edgecolor="black", linewidth=1.2,
             width=0.5, alpha=0.85)

# Add value labels
for bar, e in zip(bars, energies):
    ax.text(bar.get_x() + bar.get_width()/2, e - 8, f"{e:.1f}\neV/atom",
           ha="center", va="top", fontsize=10, fontweight="bold", color="white")

# Annotation showing the mismatch
ax.annotate("", xy=(0, -16.4), xytext=(1, -278.3),
           arrowprops=dict(arrowstyle="<->", color="red", lw=2.5))
ax.text(0.5, -150, "17× difference\nin energy scale",
       ha="center", fontsize=11, color="red", fontweight="bold",
       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red"))

ax.set_ylabel("Energy per atom (eV)", fontsize=12)
ax.set_title("Energy Scale Mismatch", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(-310, 0)

plt.tight_layout()
plt.savefig("content/assets/plots/energy_scale.png", dpi=200, bbox_inches="tight")
print("Saved: content/assets/plots/energy_scale.png")
