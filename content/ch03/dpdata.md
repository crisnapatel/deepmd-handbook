# Ch 3: Data Preparation with dpdata

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- What dpdata does: converts DFT output → DeePMD training format
- Supported formats: QE (`qe/pw-scf`), VASP (`vasp/outcar`), and more
- The raw format: `type.raw`, `box.raw`, `coord.raw`, `energy.raw`, `force.raw`
- The npy format: same data, NumPy binary (faster loading)
- Hands-on: converting VASP OUTCAR for methane (CH₄)
- Hands-on: converting QE output for graphene
- Unit pitfalls: Ry vs eV, Bohr vs Angstrom (dpdata handles this, but you need to know)
- `set_size` and splitting data into train/test sets

## Key Pitfall

> **Missing forces in QE output**: If you forgot `tprnfor = .true.` in your QE input, dpdata will silently produce garbage force data (all zeros). Always verify your forces after conversion.

> **Type map ordering**: The order in `type_map` must match your `type.raw` and must propagate consistently to param.json, POSCAR/POTCAR ordering, and pseudopotential ordering. Get this wrong and your model trains on nonsense.
