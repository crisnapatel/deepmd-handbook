# Energy Scale Traps

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

One of the most insidious pitfalls in multi-system ML potential training.

- The problem: isolated H₂ (~-16 eV/atom) vs graphene slab (~-278 eV/atom)
- Why the model can't fit both: the energy scale differs by 17x
- The loss function doesn't care about your physics — it minimizes RMSE
- Solutions:
  - Train separate models for separate systems
  - Use energy shift corrections
  - Normalize per-atom energies relative to isolated atom references
  - Restrict training to systems with comparable energy scales
- How to diagnose: `dp test` on each subsystem separately
- Real numbers from graphene + H₂ research

## Key Figure

- Bar chart: -16 vs -278 eV/atom side by side (matplotlib)
