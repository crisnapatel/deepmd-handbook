# Ch 1: Why Machine-Learned Potentials?

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- The accuracy-speed tradeoff: DFT is accurate but slow, classical force fields are fast but wrong
- Where ML potentials fit: DFT accuracy at force-field speed
- The core idea: learn the potential energy surface from DFT data
- Why DeePMD-kit specifically (end-to-end, smooth, size-extensive)
- What you'll build in this tutorial

## Key Figures

- Accuracy vs speed bubble chart (DFT, classical FF, MLIP)
- Timeline of ML potential development

## Key Pitfall

> **Systematic softening**: If you only train near equilibrium, the model underestimates forces far from equilibrium. This isn't a bug — it's a feature of incomplete training data.
