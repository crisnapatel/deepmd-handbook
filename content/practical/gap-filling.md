# Gap-Filling: Manual Data Curation

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

When dpgen's automatic active learning isn't enough — you need to manually curate training data for specific regions of configuration space.

- When gap-filling is needed: the model fails in specific conditions that dpgen doesn't explore
- The workflow: LAMMPS pre-sampling → FPS selection → QE labeling → add to training set
- Farthest Point Sampling (FPS): selecting maximally diverse configurations
- How to identify gaps: long-MD runs that crash or produce unphysical behavior
- Integrating gap-filled data back into the dpgen pipeline
- Reference: the full gap-filling workflow from real graphene + H₂ research
