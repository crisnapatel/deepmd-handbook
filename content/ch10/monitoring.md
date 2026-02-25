# Ch 10: Monitoring & Troubleshooting

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- Reading `model_devi.out`: column meanings (step, max_devi_e, min_devi_e, avg_devi_e, max_devi_f, min_devi_f, avg_devi_f)
- Plotting force deviation distributions (histogram)
- The 3 buckets in practice:
  - **Accurate** (max_devi_f < trust_lo): Model is confident and correct → skip
  - **Candidate** (trust_lo < max_devi_f < trust_hi): Model is uncertain → send to DFT
  - **Failed** (max_devi_f > trust_hi): Model is so wrong it can't learn from this → skip (for now)
- Convergence tracking: accurate fraction vs iteration number
- When to adjust trust levels
- Common errors table with real fixes:
  - DFT convergence failures
  - LAMMPS crashes (atoms too close, timestep too large)
  - Training divergence (NaN loss)
  - File permission issues
  - Disk space exhaustion

## Key Figures

- Model deviation histogram with trust_lo/trust_hi lines (matplotlib)
- Convergence plot: accurate fraction vs iteration (matplotlib)
