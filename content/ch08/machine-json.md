# Ch 8: Writing machine.json

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

machine.json tells dpgen *where* and *how* to run each stage. Three variants shown side-by-side:

### Variants

1. **Local Shell** — Everything runs on your workstation. Simplest. Good for methane tutorial.
2. **HPC (PBS/Slurm)** — Training on GPU node, exploration on GPU, DFT on CPU nodes. Real-world setup.
3. **Mixed-mode** — Training local, DFT on HPC. Useful when GPU access is limited.

### Sections covered:

1. **`train`** — Machine config for DeePMD training. GPU requirements, `group_size`.
2. **`model_devi`** — Machine config for LAMMPS exploration. GPU needed for deep potential MD.
3. **`fp`** — Machine config for DFT labeling. CPU-heavy, many tasks.
4. **`command`** — The actual executable command. The `dp train`, `lmp`, `pw.x -in input` specifics.
5. **`resources`** — Nodes, CPUs, GPUs, walltime, queue names.
6. **`module_load_path` / `module_list`** — HPC module system integration.
7. **Apptainer integration** — Container paths, bind mounts, GPU passthrough.
8. **`group_size`** — How many tasks per job submission. Too large = one failure kills all.

## Key Pitfalls

> **QE `-in input` flag**: dpdispatcher doesn't redirect stdin by default. The command must be `pw.x -in input` (or `mpirun pw.x -in input`), not just `pw.x < input`.

> **`group_size` too large**: If you set `group_size = 50` and one DFT calculation crashes, all 50 tasks in that group fail. Start with `group_size = 1` or `5`.

> **Apptainer GLIBC mismatch**: If your container was built on a different OS version, PBS/Slurm commands inside the container may fail. You need bind mounts for `/var/spool/pbs` and similar system dirs.
