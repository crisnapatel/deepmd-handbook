# Ch 2: The DeePMD Architecture

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- The descriptor: how atoms "see" their neighborhood (`se_e2_a`)
  - `rcut`, `rcut_smth`, `sel` — what they mean physically
  - Analogy: peripheral vision — how far each atom looks
- The fitting network: converting environment → energy
  - `neuron` list, `resnet_dt`, `activation_function`
- The loss function: what the model is actually optimizing
  - `start_pref_e`, `limit_pref_e`, `start_pref_f`, `limit_pref_f`
  - Why force loss dominates early, energy loss dominates late
- Learning rate schedule: exponential decay
  - `start_lr`, `stop_lr`, `decay_steps`
- Running example: real graphene training parameters

## Key Figures

- Descriptor architecture diagram (drawsvg)
- Animation: how se_e2_a builds local environment (manim)
- Loss weight evolution during training (matplotlib)

## Key Pitfall

> **Long-range interactions**: A short cutoff (`rcut = 6.0 Å`) can't capture Coulomb or dispersion interactions. For systems where these matter, you need either a larger cutoff (expensive) or specialized long-range descriptors.
