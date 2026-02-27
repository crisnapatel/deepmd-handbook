# Ch 8: Writing machine.json

You just spent all of Ch 7 telling dpgen *what* to do. Every training parameter. Every exploration temperature. Every DFT setting.

None of it matters if you can't tell dpgen *where* to do it.

That's machine.json. And I need to be honest: this file destroyed two days of my life. Not because the concepts are hard. They aren't. This is plumbing. Pure infrastructure. But it's plumbing where one wrong path, one mismatched `batch_type`, one missing bind mount will cause <mark class="silent-fail">dpgen to fail silently</mark>, submit jobs that die on arrival, or (my personal favorite) submit 200 DFT jobs to the wrong queue and incinerate your entire HPC allocation before you even notice.

Here's what nobody tells you. param.json is the brain. machine.json is the hands and feet. It tells dpgen which nodes to use, how many cores, which GPU, what scheduler to talk to, what container to run inside, what scripts to source. Get it wrong and the brain has no body. Get it right and you never think about it again.

## What machine.json Actually Does

Let's ground this before we go anywhere. dpgen has three stages per iteration: **train**, **model_devi** (exploration), and **fp** (first-principles labeling). Each stage needs to run an executable somewhere. machine.json tells dpgen, for each stage:

1. **What command to run.** `dp train`, `lmp`, `mpirun pw.x -in input`.
2. **Where to run it.** Locally via shell? Submitted to PBS/Torque? Slurm? LSF?
3. **What resources to request.** CPUs, GPUs, nodes, walltime, queue name.
4. **How to set up the environment.** Module loads, container sourcing, env vars.

Three blocks (`train`, `model_devi`, `fp`), each with the same structure: `command`, `machine`, `resources`. That's the whole file. And that's the whole trick.

```{admonition} Key Insight
:class: tip
machine.json has nothing to do with your science. Zero. It doesn't affect what the model learns or how structures are selected. It only controls the *execution infrastructure*. You could swap machine.json between a laptop and a 10,000-node supercomputer and param.json wouldn't change at all. The science lives in param.json. The logistics live here. Keep that separation clean in your head.
```

## Three Ways to Run dpgen

Before we look at the real config, let me show you the three common setups. Most people land in one of these.

**Variant 1: Local Shell.** Everything runs on your workstation or a single GPU node. Training, LAMMPS, and DFT all happen right there. No job scheduler. No queuing. This is what you use for testing your config on a small system (like Ar or water) before throwing it at the cluster. Simplest possible setup.

**Variant 2: Full HPC.** Training goes to a GPU queue via PBS/Slurm. LAMMPS exploration also goes to a GPU queue. DFT goes to a CPU queue. Every stage is a submitted job that sits in the queue, waits its turn, runs, reports back. Textbook setup for production runs when you don't have dedicated node access.

**Variant 3: Mixed-mode.** This is what I actually use. You sit on a GPU node (interactive job or dedicated allocation), run training and LAMMPS locally on that node via Shell, and submit DFT jobs to the CPU queue via PBS/Torque. GPU work happens immediately. CPU work goes through the scheduler.

Why mixed-mode? Because on my HPC, GPU nodes are scarce and the queue is brutal. Getting an interactive GPU session means I have the GPUs *right now*. Training 4 models takes 15 minutes. LAMMPS exploration takes 5 minutes. Why would I submit those to a queue and wait 2 hours for them to start? I run them locally, on my node, immediately. But DFT on 128 CPU cores? That goes to the CPU queue where nodes are plentiful and wait times are short.

This is the part the docs skip. Nobody tells you which variant to pick. So here it is, no hedging: if you can get interactive GPU access, use mixed-mode. If you can't, use Full HPC. If you're just learning, use Local Shell. Start there. Adjust later.

```{admonition} HPC Reality
:class: warning
The "right" variant depends entirely on your cluster's architecture and queue policies. Dedicated GPU node for 48 hours? Mixed-mode. Only batch submissions allowed? Full HPC. Learning dpgen on your laptop? Local Shell. There's no universal answer. Ask your sysadmin. Read your queue policies. Then pick.
```

```{figure} ../assets/diagrams/machine_modes.svg
:name: machine-modes-diagram
:width: 95%

The three machine.json execution modes compared. Local Shell runs everything on one machine. Full HPC submits all stages to the job scheduler. Mixed-mode (recommended) uses the GPU node directly for training and exploration, while submitting DFT jobs to the CPU cluster.
```

## The Real machine.json: Mixed-Mode Example

````{admonition} Real-World Research Example
:class: seealso
The machine.json walkthrough below uses a graphene + H₂ research project as the concrete example. machine.json is system-agnostic (it controls infrastructure, not science), so the same structure works for Ar, water, or any system. The paths and core counts change; the pattern doesn't.
````

Alright, enough theory. Let me show you the actual config. This is from a real research project running on an HPC with 2 GPUs and PBS/Torque for CPU queues. Not a toy example. Not a simplified version. The real thing.

```json
{
    "_comment": "HPC mixed-mode: train+model_devi on GPU node (Shell), FP via PBS to CPU nodes",
    "api_version": "1.0",
    "deepmd_version": "3.1.2",

    "train": {
        "command": "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/dp_auto_gpu.sh",
        "machine": {
            "batch_type": "Shell",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 8,
            "gpu_per_node": 2,
            "group_size": 1,
            "source_list": [
                "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/deepmd_container.sh"
            ]
        }
    },

    "model_devi": {
        "command": "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/lmp_auto_gpu.sh",
        "machine": {
            "batch_type": "Shell",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 8,
            "gpu_per_node": 2,
            "group_size": 5,
            "source_list": [
                "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/deepmd_container.sh"
            ]
        }
    },

    "fp": {
        "command": "mpirun -np 128 pw.x -nk 1 -in input",
        "machine": {
            "batch_type": "Torque",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./fp_work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 128,
            "gpu_per_node": 0,
            "queue_name": "standard",
            "group_size": 5,
            "custom_flags": [
                "#PBS -P ioe.che.jayati.1",
                "#PBS -l select=1:ncpus=128:mpiprocs=128:centos=genoa",
                "#PBS -l walltime=24:00:00"
            ],
            "source_list": [
                "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/qe_hpc.sh"
            ],
            "envs": {
                "OMP_NUM_THREADS": "1"
            }
        }
    }
}
```

72 lines. Three blocks. Let's trace through this.

## The `train` Block: Teaching the Models

The training stage runs `dp train` to build 4 DeePMD models. On our setup, this happens locally on a 2-GPU node. No scheduler. No waiting. Just go.

### The `command` Field

```json
"command": "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/dp_auto_gpu.sh"
```

Wait. Why isn't this just `"dp train"`?

Here's where it gets interesting. We have 4 models and 2 GPUs. If dpgen launches all 4 training jobs at once (and it does when `group_size` is 1), they'll all try to grab GPU 0. Three of them crash. Or worse, all four share one GPU and crawl at 25% speed each. Either way, you're not happy.

The `dp_auto_gpu.sh` script is a semaphore-based GPU assigner. It solves the problem elegantly. Open that file. I'll wait.

```bash
#!/bin/bash
# Semaphore-based GPU assignment for concurrent DeePMD training.
MAX_GPUS=2
LOCK_BASE=/tmp/dpgen_gpu

while true; do
    for gpu_id in $(seq 0 $((MAX_GPUS - 1))); do
        LOCKDIR="${LOCK_BASE}_${gpu_id}"
        if mkdir "$LOCKDIR" 2>/dev/null; then
            export CUDA_VISIBLE_DEVICES=$gpu_id
            trap "rmdir '$LOCKDIR' 2>/dev/null" EXIT INT TERM
            echo "[dp_auto_gpu] PID=$$ acquired GPU $gpu_id"
            dp "$@"
            rc=$?
            rmdir "$LOCKDIR" 2>/dev/null
            exit $rc
        fi
    done
    sleep 15   # All GPUs busy — wait and retry
done
```

Are you seeing this? Each model tries to create a lock directory (`/tmp/dpgen_gpu_0` or `/tmp/dpgen_gpu_1`). If it succeeds, it owns that GPU. If both are taken, it waits 15 seconds and tries again. When a model finishes training, it removes the lock. Models 0 and 1 run in parallel on the 2 GPUs. Models 2 and 3 wait their turn. Clean.

```{admonition} Common Mistake
:class: caution
**Do NOT use `exec dp "$@"`** inside the GPU script. I know it looks cleaner. This one will bite you. `exec` replaces the bash process, which means the `trap` handler never fires. The lock directory never gets cleaned up. All subsequent models spin-wait forever for a GPU that will never be released. I learned this the expensive way. 11 PM. My dpgen run had been "training" for 6 hours and nothing was happening. The lock was orphaned. Models 2 and 3 were just sitting there, politely waiting, forever. Don't be me.
```

If you have a simpler setup (one GPU, or you're fine with sequential training), the command can just be `"dp train input.json"`. But for multi-GPU nodes, you need something like this script.

### The `machine` Sub-block

```json
"machine": {
    "batch_type": "Shell",
    "context_type": "local",
    "local_root": "./",
    "remote_root": "./work"
}
```

Four fields, each doing one specific thing:

| Field | Value | What it does |
|-------|-------|-------------|
| `batch_type` | `"Shell"` | Run it as a shell command, right here, right now. No scheduler. No `qsub`. No `sbatch`. Alternatives: `"Torque"`, `"Slurm"`, `"LSF"`, `"PBS"` |
| `context_type` | `"local"` | Local and remote filesystems are the same machine (or share a filesystem). Alternatives: `"ssh"`, `"lazy-local"` (symlinks instead of copies) |
| `local_root` | `"./"` | Where dpgen's working directory lives. The directory where you ran `dpgen run` |
| `remote_root` | `"./work"` | Where dpdispatcher stages the actual job files. Creates a `work/` subdirectory. Files get copied here, the job runs, results get copied back |

```{admonition} Key Insight
:class: tip
The `local_root` / `remote_root` split exists because dpdispatcher was designed for remote job submission. Even when you're running locally, it still copies files from `local_root` to `remote_root`, runs the job there, and copies results back. Extra disk I/O you don't strictly need, but it keeps the architecture clean. For HPC with shared filesystems, both can be `"./"`.
```

### The `resources` Sub-block

```json
"resources": {
    "number_node": 1,
    "cpu_per_node": 8,
    "gpu_per_node": 2,
    "group_size": 1,
    "source_list": [
        "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/deepmd_container.sh"
    ]
}
```

- **`number_node`**: How many nodes per job. Training is always 1 node. DeePMD doesn't do multi-node training.

- **`cpu_per_node`**: CPUs to request. For GPU training, you don't need many. 8 is fine for data loading.

- **`gpu_per_node`**: GPUs to request. We have 2 and the GPU script handles assignment.

- **`group_size`**: Pay attention to this next part. This controls how many tasks get batched into a single job submission. For training, it's `1`. Why? Because each of the 4 models is an independent training run. With `group_size = 1`, each model gets its own job. If model 2 crashes (bad luck, GPU OOM, cosmic ray), models 0, 1, and 3 keep going. I'll do a deep dive on `group_size` shortly. It's the single most misunderstood field in this entire file.

- **`source_list`**: Shell scripts to source before running the command. This is where your environment gets set up. Here's what `deepmd_container.sh` looks like:

```bash
#!/bin/bash
# DeePMD-kit environment inside Apptainer container
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Short. Since we're already running inside an Apptainer container (we launched dpgen from within it), `dp` and `lmp` are already in the PATH. We just set a couple of environment variables and we're done.

## The `model_devi` Block: Sending the Models on a Field Trip

Exploration runs LAMMPS with all 4 frozen models to see where they disagree. Structurally, this block is almost identical to `train`. The models are the students. LAMMPS is the exam hall. The field trip is about to begin.

```json
"model_devi": {
    "command": "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/lmp_auto_gpu.sh",
    "machine": {
        "batch_type": "Shell",
        "context_type": "local",
        "local_root": "./",
        "remote_root": "./work"
    },
    "resources": {
        "number_node": 1,
        "cpu_per_node": 8,
        "gpu_per_node": 2,
        "group_size": 5,
        "source_list": [
            "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/deepmd_container.sh"
        ]
    }
}
```

Two differences from `train`:

1. **`command`** is `lmp_auto_gpu.sh` instead of `dp_auto_gpu.sh`. Same GPU-semaphore pattern, but it calls `lmp` (LAMMPS) instead of `dp`. LAMMPS with a DeePMD pair_style needs a GPU too.

2. **`group_size` is 5** instead of 1. This is the important change. Remember from Ch 6: the exploration runs a matrix of simulations. 3 systems times 3 temperatures = 9 LAMMPS jobs in iteration 0. With `group_size = 5`, dpdispatcher batches 5 of those into a single submission. So instead of 9 separate shell commands, you get 2 submissions (one with 5 tasks, one with 4).

Why 5? Each individual LAMMPS exploration is short. A few minutes on a GPU. Launching 9 separate jobs would be more overhead than the actual compute. But you also don't want `group_size = 50` because if one task fails, the entire group gets marked as failed. 5 is the middle ground. If one LAMMPS simulation blows up (the model produced insane forces and atoms went flying), you lose at most 5 tasks, not all 9. Trust me on this one.

```{admonition} Config Walkthrough
:class: note
**How `group_size` works under the hood**: dpdispatcher takes all the tasks for a stage, splits them into groups of `group_size`, and creates one job submission per group. Within each group, the tasks run *sequentially* in the same job script. So `group_size = 5` means 5 LAMMPS runs one after another inside one shell script. Not 5 in parallel.

This matters for timing. If each LAMMPS run takes 3 minutes, a group of 5 takes 15 minutes. If one of the 5 segfaults, the remaining tasks in that group might still run (dpdispatcher tries), but the group status is "failed" and dpgen may retry the whole group.
```

## The `fp` Block: Calling in DFT

This is where machine.json gets real. Training and exploration are GPU work. Fast. Lightweight. First-principles labeling is the opposite: 128 CPU cores per job, 24 hours of walltime, submitted to a PBS queue, potentially dozens of jobs per iteration.

Training and exploration are the sprinters. DFT is the marathon runner. And the marathon runner sets the pace for the whole race.

```json
"fp": {
    "command": "mpirun -np 128 pw.x -nk 1 -in input",
    "machine": {
        "batch_type": "Torque",
        "context_type": "local",
        "local_root": "./",
        "remote_root": "./fp_work"
    },
    "resources": {
        "number_node": 1,
        "cpu_per_node": 128,
        "gpu_per_node": 0,
        "queue_name": "standard",
        "group_size": 5,
        "custom_flags": [
            "#PBS -P ioe.che.jayati.1",
            "#PBS -l select=1:ncpus=128:mpiprocs=128:centos=genoa",
            "#PBS -l walltime=24:00:00"
        ],
        "source_list": [
            "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/qe_hpc.sh"
        ],
        "envs": {
            "OMP_NUM_THREADS": "1"
        }
    }
}
```

Let's trace through this piece by piece.

### The `command`: Why `-in input` Matters

```json
"command": "mpirun -np 128 pw.x -nk 1 -in input"
```

Every word matters.

- **`mpirun -np 128`**: Launch 128 MPI processes. This number must match `cpu_per_node` and the `mpiprocs=128` in the PBS flags. These three numbers must agree. Mismatch and you get idle cores (wasteful) or <mark class="silent-fail">oversubscription (wrong results, silently)</mark>.

- **`pw.x`**: Quantum ESPRESSO's main executable.

- **`-nk 1`**: Number of k-point pools. For single-point SCF on small cells (which is what dpgen fp does), 1 pool is fine.

- **`-in input`**: Read input from a file called `input`. This looks innocent. It's critical.

```{admonition} Common Mistake
:class: caution
**You MUST use `-in input`, not `< input` (stdin redirect).** Not optional. Not a suggestion.

I'm going to say that again. `-in input`. Not `< input`.

If you write `"command": "mpirun -np 128 pw.x < input"`, it won't work. dpdispatcher wraps your command in a job script and the stdin redirect doesn't propagate correctly through the script layers. The `-in input` flag tells `pw.x` to open the file directly. This is a QE-specific gotcha that doesn't exist with VASP (which reads `INCAR`/`POSCAR`/`KPOINTS` by convention).

dpgen names the QE input file `input` automatically when `fp_style` is `"qe"`. You don't choose the filename. It's always `input`.
```

### The `machine`: Torque, Not Shell

```json
"machine": {
    "batch_type": "Torque",
    "context_type": "local",
    "local_root": "./",
    "remote_root": "./fp_work"
}
```

`"batch_type": "Torque"`. Now we're talking to a scheduler. dpdispatcher generates a PBS job script, calls `qsub`, and polls `qstat` to check if the job is done. This is fundamentally different from Shell. Your dpgen process sits there, waiting, periodically poking the scheduler: "Is my job done yet?" The scheduler ignores it for a while. Then eventually it responds. Helpful.

Notice `remote_root` is `"./fp_work"` instead of `"./work"`. I keep them separate so I can easily find and clean up DFT staging files without accidentally nuking the training/exploration work directories. I learned this the expensive way.

`context_type` is still `"local"` because the GPU node and the CPU nodes share the same filesystem. dpgen writes input files locally, submits the job via `qsub`, and the compute nodes can see the files because they mount the same `/home` and `/scratch`.

### The `resources`: Everything the Scheduler Needs

```json
"resources": {
    "number_node": 1,
    "cpu_per_node": 128,
    "gpu_per_node": 0,
    "queue_name": "standard",
    "group_size": 5,
    "custom_flags": [
        "#PBS -P ioe.che.jayati.1",
        "#PBS -l select=1:ncpus=128:mpiprocs=128:centos=genoa",
        "#PBS -l walltime=24:00:00"
    ],
    "source_list": [
        "/home/chemical/phd/chz218339/scratch/hpc_jobs/env_scripts/qe_hpc.sh"
    ],
    "envs": {
        "OMP_NUM_THREADS": "1"
    }
}
```

Here's every field:

- **`cpu_per_node: 128`**: AMD EPYC Genoa node. 128 physical cores. Full node allocation.

- **`gpu_per_node: 0`**: DFT doesn't need GPUs. We're not paying the GPU premium for SCF calculations.

- **`queue_name: "standard"`**: The PBS queue to submit to. Your cluster has its own queue names: `standard`, `gpu`, `long`, `short`, whatever they decided to call them. Check with `qstat -Q` or your HPC docs.

- **`group_size: 5`**: Batch 5 DFT calculations per PBS job. This is a judgment call. Each SCF calculation on a graphene slab takes maybe 30-60 minutes on 128 cores. Five of them run sequentially in one job = roughly 5 hours per submission. With 50 candidate structures per iteration (our `fp_task_max`), that's 10 PBS jobs. Manageable.

- **`custom_flags`**: Raw PBS directives injected straight into the job script. dpdispatcher doesn't interpret them. It just pastes them into the `#PBS` header. This is your escape hatch for cluster-specific requirements that dpdispatcher doesn't natively understand.

Let me unpack each flag:

```{admonition} Config Walkthrough
:class: note
`#PBS -P ioe.che.jayati.1` -- Project/allocation code. Most HPCs require you to charge compute time to a specific project. Without this, `qsub` looks at your request and says no. Rejected. No project code, no compute.

`#PBS -l select=1:ncpus=128:mpiprocs=128:centos=genoa` -- Resource selection. One chunk, 128 CPUs, 128 MPI processes, on a CentOS Genoa-architecture node. The `centos=genoa` part is cluster-specific node filtering. Your HPC may not need it.

`#PBS -l walltime=24:00:00` -- Maximum job duration. 24 hours. If a DFT job hits this limit, PBS kills it. No mercy. Since we have `group_size = 5` and each SCF takes roughly 1 hour, 24 hours gives plenty of margin. But if your SCF calculations are bigger (larger cells, more k-points), you may need to increase this or decrease `group_size`.
```

- **`source_list`**: The QE environment setup script. Here's what `qe_hpc.sh` does:

```bash
#!/bin/bash
# QE environment for HPC (CPU-only, MPI)
ulimit -n 65536 2>/dev/null || true

INTEL_BASE=/home/soft/intel2020u4/compilers_and_libraries_2020.4.304/linux
QE_BIN=/home/apps/centos7/Quantum_Espresso/7.3.1/bin

export PATH=${QE_BIN}:${INTEL_BASE}/mpi/intel64/bin:${PATH}
export LD_LIBRARY_PATH=${INTEL_BASE}/mkl/lib/intel64_lin:...
export OMP_NUM_THREADS=2
export I_MPI_FABRICS=shm:ofi
```

No `module load` here. Why? Because we're running dpgen inside an Apptainer container, and the host's module system isn't available inside the container. So we set the paths manually. The QE binary itself (`pw.x`) lives on the host filesystem at `/home/apps/...`, and we make it accessible through bind mounts. Ugly? Yes. Reliable? Also yes. More on this in the Apptainer section below.

- **`envs`**: Environment variables set directly by dpdispatcher. `"OMP_NUM_THREADS": "1"` overrides whatever was in the source script. And yeah, you caught that. The source script sets `OMP_NUM_THREADS=2` and the envs block sets it to `1`. The `envs` block runs *after* the source scripts, so it wins. In this case, pure MPI (no OpenMP threading) turned out to work better for our cell sizes. The source script's `OMP_NUM_THREADS=2` is a leftover from standalone QE runs that I never cleaned up.

```{admonition} HPC Reality
:class: warning
**The `source_list` vs `envs` ordering matters.** dpdispatcher sources scripts first, then sets `envs` variables. If you set `OMP_NUM_THREADS=4` in your source script and `OMP_NUM_THREADS=1` in `envs`, you get 1. This is useful (intentional overrides) but also a subtle bug source if you're not aware of the ordering. When something environment-related breaks, check both places.
```

## The `group_size` Deep Dive

I keep bringing up `group_size` because it's the single most impactful parameter in machine.json for your day-to-day sanity. I cannot stress this enough. Let me give you the full picture.

`group_size` controls how many individual tasks dpdispatcher bundles into one job submission. Think of it like packing boxes for shipping. You have 50 items to send. Do you ship 50 individual packages? Or pack them 5 to a box?

**Small `group_size` (1-2):**
- Each task gets its own job. Clean isolation.
- If one fails, only that one fails.
- But you submit many jobs to the scheduler.
- Queue wait time adds up: 50 DFT jobs, each waiting 30 minutes in the queue = 25 hours of pure waiting. The scheduler is laughing at you.

**Large `group_size` (20-50):**
- Fewer total jobs submitted.
- Less queue overhead. The scheduler is happy.
- But if task 3 of 50 crashes, the whole group is in trouble.
- And one job hogs a node forever: 50 DFT jobs at 1 hour each = 50 hours of walltime. Hope you set the walltime high enough.

**The sweet spot for our graphene project:**

| Stage | `group_size` | Why |
|-------|-------------|-----|
| `train` | 1 | Only 4 tasks total. Each is independent. No reason to batch |
| `model_devi` | 5 | 9-30 short LAMMPS runs. Batching 5 reduces overhead without much risk |
| `fp` | 5 | 50 DFT jobs. 10 submissions. Each takes a few hours. If one SCF diverges, you lose 5 tasks, not 50 |

```{admonition} Common Mistake
:class: caution
**Setting `group_size = 50` for fp**: "I'll just batch everything into one job to minimize queue time!" Sounds smart. Then one SCF calculation diverges. Maybe it's a weird geometry where the H2 molecule pressed its face against the graphene sheet. The whole job script errors out. 50 DFT calculations gone. You have to delete the record.dpgen line and re-run the entire fp stage. I've seen this go wrong too many times. Start with `group_size` of 1-5 until you trust your DFT settings, then cautiously increase.
```

## The Three Variants Side by Side

Now that you understand the real mixed-mode config, let me show you how the same file looks for the other two variants. I'll focus on the structural differences.

### Variant 1: Local Shell (Everything Local)

For testing on your workstation or a single-node tutorial:

```json
{
    "api_version": "1.0",
    "deepmd_version": "3.1.2",
    "train": {
        "command": "dp",
        "machine": {
            "batch_type": "Shell",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 4,
            "gpu_per_node": 1,
            "group_size": 4
        }
    },
    "model_devi": {
        "command": "lmp",
        "machine": {
            "batch_type": "Shell",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 4,
            "gpu_per_node": 1,
            "group_size": 1
        }
    },
    "fp": {
        "command": "mpirun -np 4 pw.x -nk 1 -in input",
        "machine": {
            "batch_type": "Shell",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 4,
            "gpu_per_node": 0,
            "group_size": 1
        }
    }
}
```

Look at how clean this is. `batch_type` is `"Shell"` everywhere. No `custom_flags`, no `queue_name`, no `source_list`. Commands are bare executables (`dp`, `lmp`). `group_size = 4` for training means all 4 models run sequentially in one shell process. Fine for a single GPU. Works on your laptop. Clean.

### Variant 2: Full HPC (PBS/Slurm Everywhere)

When every stage goes through the scheduler:

```json
{
    "api_version": "1.0",
    "deepmd_version": "3.1.2",
    "train": {
        "command": "dp",
        "machine": {
            "batch_type": "Slurm",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 8,
            "gpu_per_node": 1,
            "group_size": 4,
            "queue_name": "gpu",
            "custom_flags": [
                "#SBATCH --gres=gpu:1",
                "#SBATCH --time=02:00:00",
                "#SBATCH --account=myproject"
            ],
            "source_list": ["/path/to/deepmd_env.sh"]
        }
    },
    "model_devi": {
        "command": "lmp",
        "machine": {
            "batch_type": "Slurm",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 8,
            "gpu_per_node": 1,
            "group_size": 5,
            "queue_name": "gpu",
            "custom_flags": [
                "#SBATCH --gres=gpu:1",
                "#SBATCH --time=01:00:00",
                "#SBATCH --account=myproject"
            ],
            "source_list": ["/path/to/deepmd_env.sh"]
        }
    },
    "fp": {
        "command": "mpirun -np 64 pw.x -nk 1 -in input",
        "machine": {
            "batch_type": "Slurm",
            "context_type": "local",
            "local_root": "./",
            "remote_root": "./fp_work"
        },
        "resources": {
            "number_node": 1,
            "cpu_per_node": 64,
            "gpu_per_node": 0,
            "queue_name": "cpu",
            "group_size": 5,
            "custom_flags": [
                "#SBATCH --time=24:00:00",
                "#SBATCH --account=myproject"
            ],
            "source_list": ["/path/to/qe_env.sh"],
            "envs": {
                "OMP_NUM_THREADS": "1"
            }
        }
    }
}
```

Everything is `"Slurm"`. Every stage needs `queue_name` and `custom_flags`. dpgen submits, waits, polls, collects results. The training `group_size` is 4 because with Slurm you want to minimize job submissions. GPU queue waits can be brutal. One job, 4 models sequentially, done.

The downside: every iteration involves *at least* 3 queue waits (one per stage). If your GPU queue has a 4-hour wait, that's 8+ hours of pure waiting per iteration just for train + model_devi. You're paying in wall-clock time for the convenience of not needing an interactive session. This is exactly why mixed-mode is so attractive if you can get interactive GPU access.

### Variant 3: Mixed-Mode (The Real Config)

That's the graphene config we already walked through. Train and model_devi via Shell on the GPU node. FP via Torque to the CPU queue. Best of both worlds.

The core insight: **you're already sitting on the GPU node**. You launched dpgen from within an interactive session or a dedicated allocation. Training and model_devi don't need a scheduler. They just run. Immediately. No queue. The only stage that needs the scheduler is fp, because DFT runs on different hardware (CPU nodes). That's the whole trick.

## Apptainer Integration: Running in a Container

Here's the part that had me confused for a solid day. Our DeePMD-kit, LAMMPS, and dpgen all live inside an Apptainer container. QE lives on the host. Training and exploration run inside the container. DFT runs outside it (sort of). How does any of this work together?

The approach: **launch dpgen from inside the container**, but submit PBS jobs that run on the host.

```console
$ apptainer exec --nv --writable-tmpfs \
    --bind /home:/home \
    --bind /scratch:/scratch \
    --bind /opt/pbs:/opt/pbs \
    ~/deepmd-dpgen.sif bash -c \
    'export PATH=/opt/pbs/default/bin:$PATH && dpgen run param.json machine.json'
```

Let me unpack this:

- **`--nv`**: GPU passthrough. Makes NVIDIA drivers visible inside the container. <mark class="silent-fail">Without `--nv`, `dp train` silently falls back to CPU training. 50x slower.</mark> You'll know something is wrong when training takes 8 hours instead of 15 minutes. Ask me how I know.

- **`--writable-tmpfs`**: Container filesystem is read-only by default. This gives you a writable overlay in memory so dpgen can create its working directories. Without it, random "Read-only file system" errors start popping up from Python packages trying to write `.pyc` files. Fun to debug at midnight.

- **`--bind /home:/home --bind /scratch:/scratch`**: Mount the host's `/home` and `/scratch` inside the container at the same paths. All our file paths in machine.json use host paths like `/home/chemical/phd/...`. If these aren't mounted, dpgen looks at your paths, sees nothing, panics.

- **`--bind /opt/pbs:/opt/pbs`**: This is the sneaky one. For the fp stage, dpgen calls `qsub` to submit PBS jobs. The `qsub` binary lives at `/opt/pbs/default/bin/qsub` on the host. By binding `/opt/pbs` into the container and adding it to PATH, dpgen can call `qsub` from inside the container. Without this bind mount, the fp stage fails immediately with `qsub: command not found`. The scheduler doesn't exist as far as the container is concerned.

```{admonition} HPC Reality
:class: warning
**The PBS/Slurm bind mount is the #1 source of container headaches.** On some clusters, PBS libraries live in `/var/spool/pbs` or `/opt/torque`. Slurm might need `/usr/lib/slurm` or `/run/munge`. You need to figure out where your scheduler's binaries and libraries live on the host and bind-mount all of them. If `qsub` or `sbatch` fails inside the container with a GLIBC version mismatch or missing shared library, this is almost certainly the cause.

Run `which qsub` and `ldd $(which qsub)` on the host to find all the dependencies. Then bind-mount every directory that contains a required shared library. Tedious? Yes. Necessary? Absolutely. There is no good reason to skip this step.
```

The `source_list` scripts handle the rest. Inside the container, `dp` and `lmp` are already in PATH (the container was built with them). Outside the container (for fp jobs running on CPU nodes), the `qe_hpc.sh` script sets up QE paths manually since the module system isn't available through the container's environment:

```bash
#!/bin/bash
# QE environment for HPC (CPU-only, MPI)
ulimit -n 65536 2>/dev/null || true

INTEL_BASE=/home/soft/intel2020u4/compilers_and_libraries_2020.4.304/linux
QE_BIN=/home/apps/centos7/Quantum_Espresso/7.3.1/bin

export PATH=${QE_BIN}:${INTEL_BASE}/mpi/intel64/bin:${PATH}
export LD_LIBRARY_PATH=${INTEL_BASE}/mkl/lib/intel64_lin:${INTEL_BASE}/mpi/intel64/lib:...
export I_MPI_FABRICS=shm:ofi
```

Hard-coded paths instead of `module load intel` and `module load qe`. Ugly but reliable. Modules depend on the shell environment being initialized properly, which often fails inside job scripts spawned from containers. Hard paths don't care about your shell. They just work.

```{admonition} Key Insight
:class: tip
**The fp jobs run on the host, not inside the container.** When dpdispatcher submits a PBS job via `qsub`, that job runs on whatever compute node PBS assigns. The compute node doesn't know your Apptainer container exists. It runs `qe_hpc.sh` and then `mpirun pw.x` using the host's QE installation. The container is only involved in dpgen orchestration and GPU work (training + exploration). This is why mixed-mode works so naturally with containers: the container handles the GPU/ML stack, the host handles the CPU/DFT stack. Clean separation.
```

## `remote_root`: The Disk Space Trap

Here's what nobody tells you until your `/scratch` is 98% full and the sysadmin is emailing you.

```json
"remote_root": "./work"
```

Every time dpgen runs a stage, dpdispatcher copies files into `remote_root`. For training: input JSONs, initial model files. For exploration: LAMMPS input scripts, frozen model files (4 of them, each 50-200 MB). For fp: QE input files, pseudopotential files.

Over 10 iterations with 30 LAMMPS tasks and 50 DFT tasks each, this adds up. The frozen models alone: 4 models x 150 MB x 10 iterations x 30 tasks = 180 GB. Is dpdispatcher smart enough to clean up? Sometimes. Depends on the version and whether the job completed cleanly.

````{admonition} Common Mistake
:class: caution
**Not monitoring disk usage during a dpgen run.** After 5-6 iterations, check your disk. If the `work/` or `fp_work/` directories are ballooning, you can safely delete completed iteration data from these staging directories. The *actual* results (model files, training data, model_devi.out, DFT outputs) live in the `iter.XXXXXX/` directories, not in `remote_root`. The staging directories are just scratch space.

```console
$ du -sh ./work ./fp_work

# Safe to delete after iteration is fully complete and verified
$ rm -rf ./work/iter.000000_* ./fp_work/iter.000000_*
```
````

## What Actually Happens When dpgen Runs

Let me trace through what machine.json causes at each stage of iteration 0. Not the dpgen logic (that's Ch 6 and Ch 9), but the *execution* side. The nuts and bolts of what fires when.

**Stage: train (iteration 0)**

1. dpgen creates `iter.000000/00.train/000/`, `001/`, `002/`, `003/`. One per model.
2. dpdispatcher reads the `train` block in machine.json.
3. `batch_type = Shell`. No job submission. Just run it.
4. It sources `deepmd_container.sh` (sets `OMP_NUM_THREADS=1`).
5. For each task directory, it runs the command: `dp_auto_gpu.sh train input.json`.
6. With `group_size = 1`, all 4 models launch in parallel as separate processes.
7. The GPU semaphore script assigns GPU 0 to model 0, GPU 1 to model 1. Models 2 and 3 wait.
8. Model 0 finishes (~15 min). Releases GPU 0 lock. Model 2 grabs it immediately.
9. All 4 models finish. dpdispatcher reports success.
10. Total wall time: ~30 minutes (2 batches of 2 models on 2 GPUs).

**Stage: model_devi (iteration 0)**

1. dpgen creates `iter.000000/01.model_devi/task.000.000000/` through `task.000.000008/`. 9 LAMMPS tasks (3 systems x 3 temps).
2. dpdispatcher reads the `model_devi` block.
3. `batch_type = Shell`, `group_size = 5`.
4. Group 1: tasks 0-4 run sequentially. Group 2: tasks 5-8 run sequentially.
5. Each LAMMPS run: 2-5 minutes on a GPU.
6. Total: 15-25 minutes for all 9 tasks.

**Stage: fp (iteration 0)**

1. dpgen selects candidate structures (say 45 out of the explored frames).
2. Creates `iter.000000/02.fp/task.000.000000/` through `task.000.000044/`.
3. dpdispatcher reads the `fp` block.
4. `batch_type = Torque`, `group_size = 5`.
5. Generates 9 PBS job scripts. Each script sources `qe_hpc.sh` and runs 5 sequential `mpirun pw.x` calls.
6. Submits all 9 via `qsub`. Job IDs: 12345.hpc, 12346.hpc, ...
7. Polls `qstat` every 30 seconds (dpdispatcher's default polling interval).
8. Jobs sit in queue for 0-60 minutes depending on cluster load.
9. Each job runs for roughly 5 hours (5 SCF calculations at about 1 hour each).
10. All 9 jobs finish. dpdispatcher collects outputs.
11. Total: 1-6 hours depending on queue wait and SCF convergence.

**Total iteration 0 wall time**: 2-7 hours. Training is 30 min. Exploration is 20 min. DFT is 1-6 hours. Stage 3 dominates. Always. Every single time. Check the numbers.

```{admonition} HPC Reality
:class: warning
**DFT is always the bottleneck.** In our graphene project, training takes 30 minutes, exploration takes 20 minutes, and DFT takes 4-8 hours per iteration. Over 10 iterations, that's roughly 5 hours of GPU time and 60 hours of CPU time. Plan your HPC allocation accordingly. You need a lot more CPU-hours than GPU-hours. This surprises everyone the first time.
```

## Quick Reference: Every Field in machine.json

Here's the complete field reference. Keep this open while writing your own.

```{admonition} Config Walkthrough
:class: note

| Field | Where | What it does |
|-------|-------|-------------|
| `command` | Top level of each block | The executable to run. `dp`, `lmp`, `mpirun pw.x -in input`, or a wrapper script |
| `batch_type` | `machine` | Job submission method: `Shell`, `Torque`, `Slurm`, `PBS`, `LSF` |
| `context_type` | `machine` | File transfer method: `local`, `ssh`, `lazy-local` |
| `local_root` | `machine` | Where dpgen's files live. Usually `"./"` |
| `remote_root` | `machine` | Where dpdispatcher stages job files. `"./work"` or similar |
| `number_node` | `resources` | Nodes per job. Almost always `1` |
| `cpu_per_node` | `resources` | CPUs to request |
| `gpu_per_node` | `resources` | GPUs to request. `0` for DFT |
| `group_size` | `resources` | Tasks per job submission. Start small (1-5) |
| `queue_name` | `resources` | Scheduler queue. Only for Torque/Slurm/PBS |
| `custom_flags` | `resources` | Raw `#PBS` or `#SBATCH` lines. Your escape hatch |
| `source_list` | `resources` | Shell scripts to source before running. Environment setup |
| `envs` | `resources` | Environment variables. Applied after `source_list` |
| `api_version` | Top level | dpdispatcher API version. Use `"1.0"` |
| `deepmd_version` | Top level | DeePMD-kit version. Informational |
```

## The Pitfalls Hall of Fame

Let me consolidate every trap in this file, ranked by how many hours they've personally cost me. I'm serious. Learn from my suffering.

**1. QE stdin redirect (2 days lost).** `pw.x < input` does not work through dpdispatcher. Use `pw.x -in input`. I tried every variation of quoting and escaping. Nothing works. Use the flag. Move on.

**2. Orphaned GPU locks (6 hours lost).** If your dpgen process gets killed (Ctrl+C, OOM, node reboot) while training is running, the `/tmp/dpgen_gpu_*` lock directories may survive. Next run, the models sit there spin-waiting for a GPU that will never be released. The lock's previous owner is dead, but the lock doesn't know that. Fix: `rmdir /tmp/dpgen_gpu_*` manually before restarting. Three days of compute. Gone. Because of a lock directory.

**3. GLIBC mismatch in container (1 day lost).** Container built on Ubuntu 22.04, host runs CentOS 7. `qsub` links against a newer GLIBC than the host provides. The error looks like `qsub: /lib64/libc.so.6: version 'GLIBC_2.34' not found`. The container and host are speaking different dialects. Fix: bind-mount the host's PBS/Slurm libraries into the container so it uses the host's versions.

**4. `group_size` too large (4 hours lost per incident).** One divergent SCF kills a group of 20 DFT calculations. You notice 8 hours later when you check on the run. Set it to 5 or less until your DFT settings are battle-tested.

**5. `remote_root` filling up disk (subtle, ongoing).** 200 GB of staging files accumulated over 2 weeks. Nobody noticed until the filesystem hit quota and the sysadmin sent that email. Monitor `du -sh ./work ./fp_work` periodically.

**6. `custom_flags` missing project code (instant failure).** `qsub` rejects your job with a cryptic "project not specified" error. The fix is adding `#PBS -P your.project.code` to `custom_flags`. Easy to forget when you're adapting someone else's machine.json.

**7. `mpirun -np` mismatch (wrong results or crash).** If `-np 128` doesn't match `cpu_per_node` or `mpiprocs` in the PBS flags, you either waste cores or oversubscribe. Some MPI implementations silently oversubscribe and produce wrong results instead of crashing. Silent wrong answers. The worst kind. Triple-check your numbers.

## Takeaway

machine.json is three blocks of plumbing: `train`, `model_devi`, `fp`. Each one says what command to run, where to run it, and what resources to request. The science is in param.json. The suffering is in machine.json.

Start with the Local Shell variant to test your workflow. Once it works, adapt to your cluster's scheduler and queue setup. Use the real graphene config as a template. Swap the paths, adjust the core counts, change the PBS flags to match your HPC. The structure doesn't change. Only the details do.

In Ch 9, we actually run `dpgen run param.json machine.json` and watch what happens.
