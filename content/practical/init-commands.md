# init_bulk and init_surf: Generating Initial Training Data

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

Using dpgen's initialization commands to generate initial training data before starting the active learning loop.

- `dpgen init_bulk`: for bulk crystal systems
  - Input: POSCAR + machine.json + param for init
  - What it does: generates perturbed structures → runs DFT → collects data
  - Supercell generation, perturbation magnitude, scaling factors
- `dpgen init_surf`: for surface slab systems
  - Surface construction: Miller indices, slab thickness, vacuum
  - Additional considerations for 2D systems
- When to use init vs manual data preparation
- Output format: raw/npy data ready for dpgen run
- How much initial data is "enough"? (50-200 frames as a starting point)
- Connecting init output to `init_data_sys` in param.json
