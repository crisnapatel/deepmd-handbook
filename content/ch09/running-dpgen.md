# Ch 9: Running dpgen

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

- The command: `dpgen run param.json machine.json`
- What happens when you hit Enter (the execution flow)
- Iteration directory structure:
  ```
  iter.000000/
  ├── 00.train/         # 4 model training runs
  │   ├── 000/          # Model 0
  │   ├── 001/          # Model 1
  │   ├── 002/          # Model 2
  │   └── 003/          # Model 3
  ├── 01.model_devi/    # LAMMPS exploration
  │   ├── task.000.000000/
  │   └── ...
  └── 02.fp/            # DFT labeling
      ├── task.000.000000/
      └── ...
  ```
- `dpgen.log`: what to look for
- `record.dpgen`: the state machine
  - Each line = one completed step
  - Deleting a line = re-running that step
  - Understanding step numbering (0-0, 0-1, ..., 0-5, 1-0, ...)
- Resuming after crashes
- Running inside `screen` or `tmux` (it takes hours/days)

## Key Figure

- Directory structure diagram (drawsvg)
