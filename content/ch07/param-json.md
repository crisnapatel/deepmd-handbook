# Ch 7: Writing param.json

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

This is the longest and most important chapter. We walk through every section of param.json using a real graphene + H₂ config as the running example.

### Sections covered:

1. **`type_map`** — Element ordering. Must match everywhere.
2. **`init_data_prefix` / `init_data_sys`** — Where your initial training data lives.
3. **`sys_configs`** — The 2D list gotcha: each inner list is one "system". POSCAR/structure files.
4. **`numb_models`** — Why 4 models (not 2, not 10).
5. **Training block** — `default_training_param` with descriptor, fitting_net, loss, learning_rate.
6. **`model_devi_jobs`** — The exploration schedule: temperatures, pressures, ensembles, timesteps per iteration.
7. **`model_devi_f_trust_lo` / `model_devi_f_trust_hi`** — The confidence thresholds. Setting them too tight wastes compute. Too loose wastes iterations.
8. **`fp_style`** — `"vasp"` vs `"qe"`. Differences in how params are passed.
9. **`user_fp_params`** (QE) — The full QE input. The `input_dft = 'vdw-df2-b86r'` gotcha (not `'rev-vdw-df2'`).
10. **`fp_task_max`** / `fp_task_min`** — How many DFT calculations per iteration.

### Format

Each section shown as a JSON block with line-by-line annotation. Real values from graphene param.json, with VASP equivalents in tabbed code blocks.

## Key Pitfalls

> **`sys_configs` is a 2D list**: `[["POSCAR1", "POSCAR2"], ["POSCAR3"]]` — each inner list is one thermodynamic system. A common mistake is making it 1D.

> **`input_dft` for QE 7.3.1**: Use `'vdw-df2-b86r'` not `'rev-vdw-df2'` — the latter is an obsolete label that will crash silently or produce wrong results.

> **Trust levels**: Start with `trust_lo = 0.05`, `trust_hi = 0.15` for forces (eV/Å). Adjust after seeing your first model_devi.out distribution.
