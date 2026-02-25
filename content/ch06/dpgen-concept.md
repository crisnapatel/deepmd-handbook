# Ch 6: The dpgen Concept

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- The problem: manually curating training data doesn't scale
- The solution: concurrent learning (active learning + automation)
- The 3-stage loop:
  1. **Train** (`00.train/`): Train 4 models with different random seeds
  2. **Explore** (`01.model_devi/`): Run LAMMPS MD with all 4 models, measure disagreement
  3. **Label** (`02.fp/`): Send high-disagreement frames to DFT for ground truth
- Why 4 models? Model deviation as uncertainty estimate
- The 3 buckets: accurate (< trust_lo), candidate (between), failed (> trust_hi)
- Convergence: when does the loop stop?
- `record.dpgen`: the state machine file

## Key Analogies

- Training data = curriculum for a student
- Active learning = student asking targeted questions about what they don't understand
- Model deviation = 4 students taking the same test independently; if they disagree, the question was hard
- dpgen iterations = semesters — each one the model learns from its mistakes

## Key Figures

- dpgen loop flowchart (drawsvg)
- Animation: model deviation selection process (manim)
- Animation: training set growth over iterations (manim)

## Key Pitfall

> **Model deviation ≠ true error**: 4 models can agree AND still be wrong — they just happen to all be wrong in the same way. This occurs when the system visits a region completely outside the training domain. Model deviation only catches uncertainty *within* the model's experience, not ignorance of entirely new physics.
