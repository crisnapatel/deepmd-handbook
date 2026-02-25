# Multi-Element Systems

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

Moving from single-element to multi-element systems (e.g., C+H → C+O+H).

- `type_map` ordering: must be consistent across ALL files
- `sel` for multi-element descriptors: how many neighbors of each type
- Pseudopotential ordering (QE) / POTCAR ordering (VASP): must match type_map
- `type.raw` consistency checks
- Real example: graphanol (C, H, O) param.json walkthrough
- Common failure mode: swapped element types → model trains on garbage
