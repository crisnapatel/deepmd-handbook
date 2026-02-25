# Ch 11: Validation & Production

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- `dp test` RMSE benchmarks: energy, force, virial
  - What RMSE values are "good enough"? (depends on system, property of interest)
- Long-MD stability test: run 1 ns NVT, does the system explode?
- Property validation:
  - RDF comparison (ML vs DFT-MD)
  - Diffusion coefficient
  - Equation of state
  - Phonon dispersion (if relevant)
- When is "good enough"? Match the accuracy needed for your scientific question
- Going to production: using the final model for large-scale simulations

## Key Pitfall

> **Hallucinated stability**: The model may predict a stable structure that doesn't actually exist in reality. Always validate against known experimental or high-level computational data, not just internal consistency metrics.
