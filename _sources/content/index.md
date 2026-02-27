# DeePMD + dpgen: But Make It Make Sense

```{admonition} Work in Progress
:class: warning
This tutorial is under active development. Content is being added chapter by chapter.
```

## What This Is

A conversational, from-scratch tutorial for building machine-learned interatomic potentials with [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) and automating the training pipeline with [dpgen](https://github.com/deepmodeling/dpgen).

## Who This Is For

- Graduate students starting with ML potentials
- Researchers who know DFT (QE or VASP) but not ML
- Anyone who found the official docs... dense

**Prerequisites**: Basic Linux command line, basic MD concepts (NVT/NPT, timestep), basic QE or VASP familiarity. No ML knowledge assumed. The tutorial covers that from the ground up.

## Software Versions

All tutorials use the following versions inside an Apptainer container:

| Software | Version |
|----------|---------|
| DeePMD-kit (`dp`) | v3.1.2 |
| dpdata | 1.0.0 |
| dpgen | 0.13.2 |
| LAMMPS (`lmp`) | 29 Aug 2024 |
| Python | 3.12.12 |

Container: `deepmd-dpgen.sif`

## Highlights

Throughout this tutorial, you'll see color-coded highlights on sentences that really matter:

- <mark class="silent-fail">Yellow</mark> — **Silent failures.** Things that break without any error or warning. The model trains, the loss converges, the physics is wrong. These are the traps.
- <mark class="hard-req">Pink</mark> — **Hard requirements.** Non-negotiable. Not a suggestion, not a best practice. Get this wrong and nothing works.
- <mark class="key-insight">Green</mark> — **Key insights.** The "aha" moments. Core ideas that tie everything together and shift how you think about the problem.

## How to Read This

See the {ref}`reading-order` page for suggested paths through the material depending on your background.

## Hands-On Examples

This tutorial uses multiple systems of increasing complexity:

1. **Methane (CH₄)**: The "hello world". 5 atoms, ~20 min training. Learn dpdata, dp train, dp freeze, dp test, LAMMPS MD.
2. **Methane dpgen loop**: Same system, now with dpgen. Learn param.json, machine.json, the 3-stage active learning loop.
3. **Bulk water (H₂O)**: Condensed phase with hydrogen bonding. Learn multi-frame systems, NVT/NPT exploration, property validation.
4. **Simple metal or Ar**: Single-element system. Learn init_bulk, crystal structures, EOS validation.
5. **Graphene + H₂ (C+H)**: Real research system. Slab + adsorbate, energy scale issues, gap-filling, QE as fp_style.

## Citation

If you find this tutorial useful, please cite the underlying software:

- Zhang et al., "DP-GEN: A concurrent learning platform...", *Comput. Phys. Commun.* (2020)
- Zeng et al., "DeePMD-kit v2...", *J. Chem. Phys.* (2023)
