#!/usr/bin/env python3
"""SOAP descriptor PCA: colored by FCC/liquid (Ar) or dataset (water).
Two-panel figure: left=Ar, right=Water.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dscribe.descriptors import SOAP
from ase import Atoms
import os

out_path = os.path.join(os.path.dirname(__file__), "..", "content", "assets", "plots", "soap_pca.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- Ar SOAP PCA ---
ar_root = "/home/krishna/scratch/qe-dpdata-dpgen/ar_deepmd/01_data"
soap_ar = SOAP(species=["Ar"], r_cut=6.0, n_max=4, l_max=4, average="outer", periodic=True)

ar_descs = []
ar_labels = []

for label, phase in [("FCC", "ar_fcc"), ("Liquid", "ar_liquid")]:
    base = os.path.join(ar_root, "training", phase, "set.000")
    coords = np.load(os.path.join(base, "coord.npy"))
    boxes = np.load(os.path.join(base, "box.npy"))

    for i in range(len(coords)):
        pos = coords[i].reshape(-1, 3)
        box = boxes[i].reshape(3, 3)
        atoms = Atoms("Ar" * len(pos), positions=pos, cell=box, pbc=True)
        desc = soap_ar.create(atoms)
        ar_descs.append(desc.flatten())
        ar_labels.append(label)

ar_descs = np.array(ar_descs)
pca = PCA(n_components=2)
ar_pca = pca.fit_transform(ar_descs)

for label, color, marker in [("FCC", "#2196F3", "o"), ("Liquid", "#FF5722", "s")]:
    mask = np.array(ar_labels) == label
    axes[0].scatter(ar_pca[mask, 0], ar_pca[mask, 1], s=30, alpha=0.7,
                    color=color, marker=marker, edgecolors="white", linewidth=0.5, label=label)

axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
axes[0].set_title("Ar: FCC vs Liquid", fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# --- Water SOAP PCA ---
water_root = "/home/krishna/scratch/qe-dpdata-dpgen/water_deepmd/00_data"

# Read type_map
with open(os.path.join(water_root, "training", "data_0", "type_map.raw")) as f:
    type_map = f.read().split()
with open(os.path.join(water_root, "training", "data_0", "type.raw")) as f:
    types = [int(x) for x in f.read().split()]

soap_water = SOAP(species=type_map, r_cut=5.0, n_max=4, l_max=4, average="outer", periodic=True)

water_descs = []
water_labels = []

datasets = [("training", "data_0"), ("training", "data_1"), ("training", "data_2"), ("validation", "data_3")]
for split, dname in datasets:
    base_dir = os.path.join(water_root, split, dname)
    for sdir in sorted(os.listdir(base_dir)):
        if not sdir.startswith("set."):
            continue
        spath = os.path.join(base_dir, sdir)
        c_file = os.path.join(spath, "coord.npy")
        b_file = os.path.join(spath, "box.npy")
        if not os.path.exists(c_file):
            continue
        coords = np.load(c_file)
        boxes = np.load(b_file)

        # Sample at most 20 frames per set for speed
        n_sample = min(20, len(coords))
        indices = np.linspace(0, len(coords) - 1, n_sample, dtype=int)

        for i in indices:
            pos = coords[i].reshape(-1, 3)
            box = boxes[i].reshape(3, 3)
            symbols = [type_map[t] for t in types]
            atoms = Atoms(symbols, positions=pos, cell=box, pbc=True)
            desc = soap_water.create(atoms)
            water_descs.append(desc.flatten())
            water_labels.append(dname)

water_descs = np.array(water_descs)
pca2 = PCA(n_components=2)
water_pca = pca2.fit_transform(water_descs)

cmap = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
for idx, dname in enumerate(["data_0", "data_1", "data_2", "data_3"]):
    mask = np.array(water_labels) == dname
    lbl = f"{dname} ({'val' if dname == 'data_3' else 'train'})"
    axes[1].scatter(water_pca[mask, 0], water_pca[mask, 1], s=30, alpha=0.7,
                    color=cmap[idx], edgecolors="white", linewidth=0.5, label=lbl)

axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.0%})")
axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.0%})")
axes[1].set_title("Water: dataset coverage", fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.2)

fig.suptitle("SOAP descriptor PCA: configuration space coverage", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
