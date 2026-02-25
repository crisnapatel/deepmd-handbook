# Ch 4: Your First DeePMD Model

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- The input JSON for `dp train`: descriptor, fitting_net, loss, learning_rate, training
- Walking through a minimal input.json for methane
- Running training: `dp train input.json`
  - Inside Apptainer: `apptainer exec deepmd-dpgen.sif dp train input.json`
- Reading `lcurve.out`: what each column means
  - Plotting the learning curve
  - How to tell if training is converging vs overfitting
- Freezing the model: `dp freeze -o graph.pb`
- Testing the model: `dp test -m graph.pb -s test_data/ -n 50`
  - RMSE interpretation: what's "good enough"?

## Key Figures

- Learning curve plot from lcurve.out (matplotlib)
- dp test output interpretation

## Key Pitfall

> **Overfitting small datasets**: With only 50-100 frames, a large network will memorize instead of generalize. Start small (2-3 layers of 25 neurons) and increase only if underfitting.
