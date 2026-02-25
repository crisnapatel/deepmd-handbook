# Apptainer Setup

```{admonition} Status
:class: warning
Stub — content coming soon.
```

## Outline

Running DeePMD-kit, LAMMPS, and dpgen inside Apptainer containers on HPC.

- Why containers: reproducibility, dependency hell avoidance
- Building the container: `apptainer build deepmd-dpgen.sif docker://...`
- Bind mounts: what directories need to be visible inside the container
  - Home directory, scratch, data directories
  - PBS/Slurm system directories (`/var/spool/pbs`, etc.)
- GPU passthrough: `--nv` flag
- The GLIBC compatibility issue: when your container OS is too old/new for the host
- Running commands: `apptainer exec [--nv] [--bind ...] deepmd-dpgen.sif <command>`
- Integration with dpgen's machine.json: setting `singularity_image` or wrapping commands
- Debugging: common container errors and fixes
