# QE vs VASP for First-Principles Labeling

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

How param.json and machine.json differ when using Quantum ESPRESSO vs VASP as the DFT engine.

- `fp_style`: `"vasp"` vs `"qe"`
- param.json differences:
  - VASP: `user_incar_params`, POTCAR handling, KPOINTS
  - QE: `user_fp_params` with full `&CONTROL`, `&SYSTEM`, `&ELECTRONS` namelists
- The QE `input_dft` gotcha: `'vdw-df2-b86r'` not `'rev-vdw-df2'` on QE 7.3.1
- machine.json differences:
  - VASP: `vasp_std` or `vasp_gam` command
  - QE: `pw.x -in input` (NOT stdin redirect)
- Pseudopotential handling:
  - VASP: POTCAR concatenation by type_map order
  - QE: `pseudo_dir` + per-species `.upf` files
- Performance considerations: QE is free, VASP needs a license
- Converting between the two: what changes, what stays the same
