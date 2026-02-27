# Apptainer Setup

I spent two weeks trying to compile DeePMD-kit from source on the HPC. Two weeks. Tracking down the right TensorFlow version. Rebuilding LAMMPS with the DeePMD plugin. Coaxing MPI into cooperating with CUDA. Wrestling with `cmake` flags that the documentation swore were correct but produced linking errors that looked like ancient curses. I got it working exactly once. Then a system update broke everything.

Don't be me.

A container is a frozen filesystem image. Everything your code needs (binaries, libraries, Python packages, CUDA runtimes) packed into a single `.sif` file. You copy that file to any HPC cluster, and it runs. No module juggling. No version conflicts. No "it works on my machine" followed by three hours of debugging why it doesn't work on yours. Think of it as a pre-packaged lab. The HPC's ancient system libraries, the weird module conflicts, the missing headers? None of that matters. Your software lives inside the container, hermetically sealed.

For DeePMD-kit specifically, this is salvation. The dependency tree is vicious: TensorFlow or PyTorch (pinned to a specific version), CUDA toolkit (must match both TF/PyTorch and your GPU driver), MPI (must match the host interconnect), LAMMPS (compiled with the DeePMD plugin), dpgen, dpdata, and a dozen Python packages that all need compatible versions. Getting all of those aligned by hand is an exercise in suffering. I know because I did it. Twice. The second time was not faster.

A container bundles all of it. Pull it, run it, move on with your actual research.

---

## What's in Our Container

The exact versions used throughout this tutorial:

| Package | Version |
|---------|---------|
| DeePMD-kit (`dp`) | 3.1.2 |
| dpdata | 1.0.0 |
| dpgen | 0.13.2 |
| LAMMPS | 29 Aug 2024 |
| Python | 3.12.12 |
| CUDA | 12.1 |

These versions are known to work together. That sentence alone is worth the container. If you have never spent a day discovering that your TensorFlow build expects CUDA 11.8 while your cluster only has CUDA 12.0, you will not fully appreciate this. If you have, you are already downloading the `.sif`.

---

## Getting the Container

### Option 1: Pull from Docker Hub

One command:

```console
$ apptainer pull deepmd-dpgen.sif docker://deepmdkit/deepmd-kit:3.1.2_cuda12.1_gpu
```

Apptainer grabs the Docker image and converts it to its `.sif` format. The file will be 2-5 GB depending on what is bundled. Make sure you have the space before you start.

```{admonition} HPC Reality
:class: warning
Many HPC login nodes have tight disk quotas in `$HOME`. Pull the `.sif` into your scratch or project directory, not your home. Also: some clusters block Docker Hub pulls from compute nodes. Do this on a login node or a data transfer node. I learned this the expensive way, watching a pull timeout after 20 minutes on a compute node that had no outbound network access. The error message was not helpful.
```

### Option 2: Build from a Definition File

If you need to customize (add packages, change versions, include your own scripts), write a `.def` file and build:

```console
$ apptainer build deepmd-dpgen.sif deepmd-dpgen.def
```

Building from source takes 20-60 minutes and requires root or `--fakeroot` privileges. Most HPC clusters allow `--fakeroot`. If yours doesn't, build on a local machine and `scp` the `.sif` over.

For this tutorial, pulling from Docker Hub is all you need.

---

## Running Commands Inside the Container

The pattern is always the same: `apptainer exec <image> <command>`. You run a command, it executes inside the container's filesystem, and it exits. No persistent shell. No daemon. No overhead worth worrying about.

```console
$ apptainer exec deepmd-dpgen.sif dp --version          # Check DeePMD-kit version
$ apptainer exec deepmd-dpgen.sif lmp -h                 # Check LAMMPS + installed packages
$ apptainer exec --nv deepmd-dpgen.sif dp train input.json   # Run dp train with GPU
$ apptainer exec --nv deepmd-dpgen.sif lmp -in in.lammps     # Run LAMMPS with DeePMD potential
```

```{admonition} Config Walkthrough
:class: note
The pattern is always `apptainer exec [flags] <image> <command> [args]`. The `--nv` flag enables GPU passthrough. Include it for any GPU workload (`dp train`, `lmp` with a DeePMD potential). Omit it for CPU-only tasks like `dp test` on small systems or data conversion scripts.
```

Every command you would normally type at the terminal, you prefix with `apptainer exec [flags] deepmd-dpgen.sif`. That is the whole mental model. Container runs your command, produces output, disappears. Clean.

---

## Bind Mounts: Making Your Files Visible

Here is where people get stuck. And I mean everyone. I got stuck here. My labmates got stuck here. The postdoc who showed me Apptainer in the first place got stuck here when he moved to a new cluster.

<mark class="hard-req">Apptainer sees its own filesystem. By default, it has no idea your scratch directory exists.</mark> Your data? Invisible. Your config files in some project directory? Doesn't know about them. The container is a sealed box, and you need to punch holes in it.

Those holes are called **bind mounts.**

```console
$ apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind /home:/home \
    deepmd-dpgen.sif dp train input.json
```

The syntax is `--bind /host/path:/container/path`. If you leave off the container path, it maps to the same path inside. So `--bind /scratch:/scratch` makes the host's `/scratch` appear at `/scratch` inside the container. Same path, same files, no confusion.

```{admonition} HPC Reality
:class: warning
Critical directories to bind:

- **`/home`**: your scripts, configs, maybe input data
- **`/scratch` or `/work`**: where your actual data and jobs live
- **`/var/spool/pbs`**: on PBS/Torque systems, this is where job environment variables live. Without this bind, your PBS job scripts can't read `$PBS_NODEFILE` or `$PBS_JOBID` from inside the container. Your job launches, the container can't find the node file, MPI can't figure out where to run, everything falls apart. Ask me how I know.
- **Any project or group directories**: `/project`, `/groups`, whatever your cluster uses

If a path doesn't exist inside the container, Apptainer creates it automatically as a bind point. No need to pre-create anything.
```

Typing `--bind` flags every single time gets old by the third command. Set `APPTAINER_BIND` in your shell profile and forget about it:

```bash
export APPTAINER_BIND="/scratch:/scratch,/home:/home,/var/spool/pbs:/var/spool/pbs"
```

Now every `apptainer exec` inherits those mounts automatically. Set it once, never think about it again. There is no good reason to skip this step.

---

## GPU Passthrough

The `--nv` flag is what gives the container access to your NVIDIA GPUs. Without it, `dp train` runs on CPU. <mark class="silent-fail">It won't crash. It won't complain. It will just silently be 10-100x slower</mark>, and you will sit there watching training steps tick by at a glacial pace wondering what you did wrong.

Nothing. You just forgot two characters: `--nv`.

I'm serious. This one will bite you. I have watched three different people make this exact mistake, stare at the training log for 30 minutes, and then realize the GPU was never engaged. One of them let it run overnight on CPU before noticing.

```{admonition} Common Mistake
:class: caution
Forgot `--nv` in your job script? DeePMD-kit happily trains on CPU without a single warning. The only clue is the training log: no GPU device mentioned, and each step takes forever. Always verify GPU access before launching a multi-hour training run. The 10 seconds it takes to check will save you hours.
```

Verify GPU access like this:

```console
$ apptainer exec --nv deepmd-dpgen.sif python -c \
    "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see something like:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If you see an empty list `[]`, the GPU is not accessible. Three things to check:
1. You included `--nv`
2. The host has NVIDIA drivers loaded (`nvidia-smi` on the host should work)
3. Your job actually landed on a GPU node (not a CPU-only node)

That third one bites people more than you'd think. You requested a GPU node, but the scheduler gave you a CPU node because the GPU queue was full and your script didn't enforce it. Check your PBS/Slurm flags.

---

## The GLIBC Compatibility Issue

This one is insidious. You pull a container, it works perfectly on one cluster. You copy it to another cluster. And then:

```
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
```

Here is what is happening: Apptainer bind-mounts some host libraries (especially for GPU and MPI) into the container. If the container's base OS is much older or newer than the host OS, the GLIBC versions clash. The host's Slurm or PBS libraries expect a GLIBC version that doesn't exist in the container. Or vice versa. Either way, nothing works. And the error message tells you exactly what went wrong but gives you zero guidance on how to fix it.

```{admonition} HPC Reality
:class: warning
Fixes, in order of preference:

1. **Use a container built on a compatible OS.** If your cluster runs RHEL 8, use a container based on Ubuntu 20.04+ or RHEL 8+. The DeePMD Docker images are typically based on recent Ubuntu, which works on most modern clusters.
2. **Bind-mount the host's libc into the container.** Ugly but effective: `--bind /lib64:/host-lib64` and set `LD_LIBRARY_PATH` inside the container. This is fragile and version-specific. It feels like a hack because it is one.
3. **Use `--contain` or `--cleanenv`** to prevent the host's environment from leaking into the container. This can help with library conflicts but may break MPI.

If you hit GLIBC errors, the first question is: what OS is the container based on, and what OS is the host running? `cat /etc/os-release` on both sides will tell you instantly.
```

---

## Integration with dpgen's machine.json

When dpgen runs, it submits training, exploration, and labeling jobs to your HPC scheduler. It needs to know how to invoke `dp`, `lmp`, and `pw.x` (or `vasp`). With a container, you have two approaches.

### Approach 1: Wrap the Command Directly

In your `machine.json`, set the command field to include the full `apptainer exec` invocation:

```json
{
    "command": "apptainer exec --nv --bind /scratch:/scratch /path/to/deepmd-dpgen.sif dp",
    "machine_type": "slurm"
}
```

Every time dpgen invokes `dp`, it actually runs `apptainer exec ... dp`. Same for LAMMPS:

```json
{
    "command": "apptainer exec --nv --bind /scratch:/scratch /path/to/deepmd-dpgen.sif lmp"
}
```

Explicit. Portable. Anyone reading the config knows exactly what is happening. No guessing. I prefer this approach. When something breaks at 2 AM and you are reading `machine.json` through bleary eyes, you want the command to be right there, spelled out, no indirection.

### Approach 2: Source Script

Write a wrapper script that sets up the environment and call it via `source_list` or `module_list` in your machine.json:

```bash
#!/bin/bash
# setup_deepmd.sh
export DEEPMD_SIF="/scratch/containers/deepmd-dpgen.sif"
alias dp="apptainer exec --nv --bind /scratch:/scratch $DEEPMD_SIF dp"
alias lmp="apptainer exec --nv --bind /scratch:/scratch $DEEPMD_SIF lmp"
```

````{admonition} Config Walkthrough
:class: note
Approach 1 is more reliable for dpgen. Source scripts with aliases can behave differently depending on whether the shell is interactive or non-interactive. Job scripts are non-interactive. Aliases may not expand.

If you use Approach 2, use shell functions instead of aliases, or export the full path and use variable expansion in `command`. I have been burned by aliases silently not expanding in PBS scripts. Spent an hour wondering why dpgen couldn't find `dp`. The answer was that `dp` was an alias, and bash ignores aliases in non-interactive mode. Nothing crashed. dpgen just reported "command not found" in a log file buried three directories deep.

```bash
# Use functions instead of aliases for non-interactive shells:
dp() { apptainer exec --nv --bind /scratch:/scratch $DEEPMD_SIF dp "$@"; }
export -f dp
```
````

---

## Common Container Errors

Let me save you some debugging time. These are errors I have actually hit, with the actual fixes.

**1. Permission denied / Read-only filesystem**

```
OSError: [Errno 30] Read-only filesystem: '/some/path'
```

The `.sif` image is immutable. If your code tries to write to a path inside the container (not on a bind-mounted directory), it fails hard. The container is read-only by design.

Fix:

```console
$ apptainer exec --writable-tmpfs --nv deepmd-dpgen.sif dp train input.json
```

`--writable-tmpfs` creates a temporary writable overlay in RAM. Writes go to tmpfs and disappear when the container exits. Good for temp files. Bad for large outputs, because they eat your RAM.

**2. Out of disk space in /tmp**

DeePMD-kit and TensorFlow love writing temp files. If `/tmp` inside the container is small (it often is), training crashes mid-run. Halfway through. After you already waited two hours. I cannot stress this enough: bind-mount a larger tmp.

```console
$ apptainer exec --nv --bind /scratch/tmp:/tmp deepmd-dpgen.sif dp train input.json
```

Or set `TMPDIR`:

```console
$ export APPTAINER_TMPDIR=/scratch/tmp
```

**3. Module not found / ImportError**

```
ModuleNotFoundError: No module named 'dpdata'
```

You are accidentally running the host's Python, not the container's. This happens when the host has a Python in `$PATH` that shadows the container's Python. The host Python has no idea what dpdata is. It is not supposed to.

Use `--cleanenv` to prevent host environment leakage:

```console
$ apptainer exec --cleanenv --nv deepmd-dpgen.sif python -c "import dpdata"
```

**4. MPI errors with multi-node jobs**

Running across multiple nodes requires the host's MPI to communicate with the container's MPI. This is the hardest problem in containerized HPC. The versions must be ABI-compatible. If they are not, you get segfaults or MPI initialization failures that produce the most cryptic error messages you have ever seen.

For single-node GPU training (the common case for DeePMD-kit), this is not your problem. For multi-node LAMMPS exploration runs, you may need to bind-mount the host's MPI libraries or use a container built with the same MPI as the host.

```{admonition} Common Mistake
:class: caution
Don't overthink MPI compatibility for `dp train`. DeePMD-kit training typically runs on a single GPU. You don't need multi-node MPI for training. The MPI headache only matters for large-scale LAMMPS exploration runs, and even those often fit on a single node. Solve the MPI problem when you actually have it, not before.
```

---

## Quick Reference

```console
$ apptainer pull deepmd-dpgen.sif docker://deepmdkit/deepmd-kit:3.1.2_cuda12.1_gpu

$ apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind /home:/home \
    deepmd-dpgen.sif dp train input.json

$ apptainer exec --nv deepmd-dpgen.sif python -c \
    "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

$ apptainer shell --nv --bind /scratch:/scratch deepmd-dpgen.sif
```

Two commands to pull. One pattern to run everything. The rest of this tutorial assumes you have the container available and know how to invoke it. Every `dp`, `lmp`, and `python` command in subsequent chapters should be prefixed with the appropriate `apptainer exec` invocation, or you have set up `APPTAINER_BIND` and `--nv` so the container is transparent.

That is containers. Not glamorous. Not the reason you got into computational science. Just a thing that works so you can focus on the part that actually matters: your science.
