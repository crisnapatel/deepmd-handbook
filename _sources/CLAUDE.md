# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conversational tutorial website for the DeePMD-kit + dpgen machine learning interatomic potential pipeline. Built with Jupyter Book (MyST Markdown), deployed to GitHub Pages at `crisnapatel.github.io/deepmd-dpgen-tutorial/`.

Target audience: grad students who know Linux, QE/VASP, and basic MD, but NOT machine learning or dpgen.

## Build & Deploy

```bash
# Build locally (requires Analysis conda env or jupyter-book installed)
conda run -n Analysis jupyter-book build .

# Serve locally for review
python -m http.server 8888 --directory _build/html

# Deploy: push to main triggers .github/workflows/deploy.yml
# which builds and publishes to gh-pages branch via peaceiris/actions-gh-pages
```

## Content Structure

- `content/` — All tutorial chapters and practical topics (MyST Markdown `.md` files)
  - `ch01/`–`ch11/` — Main chapters (Part I: Foundations, Part II: First Model, Part III: Active Learning, Part IV: Validation)
  - `practical/` — 7 standalone deep-dive pages (gap-filling, multi-element, energy-scale, transfer-learning, apptainer, qe-vs-vasp, init-commands)
  - `assets/plots/` — matplotlib-generated PNGs
  - `assets/diagrams/` — drawsvg-generated SVGs
- `gen_figures/` — Python scripts that generate all visual assets
  - `gen_*.py` — matplotlib plots → `content/assets/plots/`
  - `draw_*.py` — drawsvg diagrams → `content/assets/diagrams/`
- `_config.yml` — Jupyter Book config (MyST extensions, repository metadata)
- `_toc.yml` — Table of contents defining book structure

## Figure Generation

All figures are programmatically generated. To regenerate:

```bash
cd gen_figures
# matplotlib plots (need matplotlib)
conda run -n Analysis python gen_accuracy_speed.py
conda run -n Analysis python gen_lcurve.py
# ... etc

# SVG diagrams (need drawsvg)
conda run -n Analysis python draw_dpgen_loop.py
conda run -n Analysis python draw_deepmd_architecture.py
# ... etc
```

Scripts output directly to `content/assets/plots/` or `content/assets/diagrams/`. Run from repo root or `gen_figures/` directory.

## Writing Style

Content follows the `conversational-dpgen` skill. Key rules:
- **Reader-centric voice**: "you", "we" (shared discovery). Never "we teach" or professorial framing
- **Minimal em-dashes**: Use periods, semicolons, or parentheses instead
- **MyST admonitions**: `{admonition}` blocks with classes: `tip` (Key Insight), `warning` (HPC Reality), `note` (Config Walkthrough), `danger` (Warning: Energy Scale), `caution` (Common Mistake)
- **Figures**: Use `{figure}` directive with `:name:` and `:width:` attributes
- **Comparison tables**: Use tables when discussing parallel concepts (DFT vs FF vs MLIP, etc.)
- **JSON walkthroughs**: Show real config snippets; Ar and water as primary examples, graphene+H₂ in research sidebars
- **Figures**: Include Ar/water data-driven plots (parity, lcurve, RDF, thermo) alongside conceptual diagrams

## Primary Tutorial Examples

- **Argon (Ar)**: Single-element "hello world". 32 atoms, FCC + liquid, 200 frames, PBE+D3, virial included. E 0.3 meV/atom, F 3-5 meV/Å
- **Water (H₂O)**: Multi-element example. 192 atoms (64 H₂O), 320+80 frames from ICTP 2024, no virial. E 0.43 meV/atom, F 38.5 meV/Å
- **Graphene + H₂**: Research-scale example, appears in Ch 7 param.json walkthrough and practicals. NOT the primary teaching example

## Key Conventions

- Software versions pinned: DeePMD-kit v3.1.2, dpdata 1.0.0, dpgen 0.13.2, LAMMPS 29 Aug 2024
- All examples run inside Apptainer container (`deepmd-dpgen.sif`)
- Primary DFT code is QE (Quantum ESPRESSO), VASP shown in tabbed code blocks
- QE gotcha: `input_dft = 'vdw-df2-b86r'` not the obsolete `'rev-vdw-df2'`
- The isolated H₂ energy (~-16 eV/atom) vs slab energy (~-278 eV/atom) mismatch is a known trap covered in `practical/energy-scale.md`
- `type_map` element ordering must match across param.json, type.raw, type_map.raw, POSCAR, pseudopotentials, and `sel` array
