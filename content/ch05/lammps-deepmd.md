# Ch 5: LAMMPS with Deep Potential

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- The `pair_style deepmd` command
- Minimal LAMMPS input script for MD with a deep potential
- NVT simulation: thermostat, timestep, dump frequency
- Energy conservation check (NVE): does the model conserve energy?
- RDF comparison: deep potential vs DFT reference
- Running inside Apptainer with GPU

## Key Concepts

- `pair_style deepmd graph.pb` — that's it, one line
- LAMMPS atom types must match your type_map ordering
- Timestep considerations for ML potentials
