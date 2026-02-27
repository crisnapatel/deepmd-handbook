# Ch 7: Writing param.json

I'm going to walk you through the file that controls everything.

Not "most things." Not "the important parts." Everything. Which elements you're studying. How the neural network is shaped. What temperatures to explore. When to call DFT. How to grade the results. Every scientific decision in your dpgen run lives inside `param.json`.

The dpgen docs list every parameter with one-sentence descriptions that explain nothing. You copy them in, change two values, hold your breath, and submit. That's how I started. It's also how I burned through three failed dpgen runs before I understood what any of it actually controlled. Three runs. Weeks of HPC allocation. Because I treated `param.json` like a form to fill out instead of a blueprint to understand.

So here's what this chapter is. First, I'll show you the standalone training configs we used for our Ar and water tutorial models. Simple, concrete, no dpgen yet. Then, a real `param.json` from a graphene + H₂ research project (the full dpgen workflow), walked through section by section, field by field. What it does, why it's set to that value, and what happens when you get it wrong.

## Quick Reference: Standalone Training Configs

Before diving into dpgen's `param.json`, let's look at the DeePMD `input.json` files we used in Ch 4. These are the configs you use with `dp train`, not with `dpgen run`. They teach the same concepts.

### Ar input.json (Single Element)

```json
{
  "model": {
    "type_map": ["Ar"],
    "descriptor": {
      "type": "se_e2_a",
      "rcut": 9.0,
      "rcut_smth": 2.0,
      "sel": [80],
      "neuron": [10, 20, 40],
      "axis_neuron": 8
    },
    "fitting_net": {
      "neuron": [60, 60, 60],
      "resnet_dt": true
    }
  },
  "loss": {
    "start_pref_e": 0.02, "limit_pref_e": 2.0,
    "start_pref_f": 1000, "limit_pref_f": 1.0,
    "start_pref_v": 0.02, "limit_pref_v": 1.0
  },
  "training": {
    "training_data": {
      "systems": ["../01_data/training/ar_fcc", "../01_data/training/ar_liquid"]
    },
    "validation_data": {
      "systems": ["../01_data/validation/ar_fcc", "../01_data/validation/ar_liquid"]
    },
    "numb_steps": 400000
  }
}
```

One element. One `sel` value. Virial training enabled (we have stress from DFT). 200 frames, 400k steps, ~30 min on GPU. Result: 0.3 meV/atom energy, 3-5 meV/Angstrom forces.

### Water input.json (Multi-Element)

```json
{
  "model": {
    "type_map": ["O", "H"],
    "descriptor": {
      "type": "se_e2_a",
      "sel": [46, 92],
      "rcut": 6.00,
      "neuron": [25, 50, 100],
      "axis_neuron": 16
    },
    "fitting_net": {
      "neuron": [240, 240, 240],
      "resnet_dt": true
    }
  },
  "loss": {
    "start_pref_e": 0.02, "limit_pref_e": 1,
    "start_pref_f": 1000, "limit_pref_f": 1,
    "start_pref_v": 0, "limit_pref_v": 0
  },
  "training": {
    "training_data": {
      "systems": ["../00_data/training/data_0/", "../00_data/training/data_1/", "../00_data/training/data_2/"]
    },
    "validation_data": {
      "systems": ["../00_data/validation/data_3"]
    },
    "numb_steps": 500000
  }
}
```

Two elements. Two `sel` values: 46 O neighbors, 92 H neighbors (within 6 Angstroms). No virial (ICTP data lacks stress tensors). Bigger network (multi-element needs more capacity). Result: 0.43 meV/atom energy, 38.5 meV/Angstrom forces.

Notice the pattern: `sel` has one entry per element type, always matching `type_map` order. The descriptor network scales with `sel`. The fitting network scales with system complexity. Virial is on or off based on what your DFT data provides.

---

Now let's look at a full dpgen `param.json`.

Pull up your param.json.

## The Big Picture

`param.json` is the brain of your dpgen run. Not the muscles (that's `machine.json`). Not the skeleton (that's your directory structure). The brain. Every decision flows from this file.

- **What** you're training (element types, network architecture, loss function)
- **Where** the initial data lives (the seed dataset)
- **How** to explore (temperatures, pressures, timesteps, which structures)
- **When** to label (trust levels, task limits, convergence thresholds)
- **What DFT engine** to use and with what settings

`machine.json` (Ch 8) handles execution logistics: which queue, how many cores, container paths. `param.json` handles the science.

Think of it this way. `param.json` is the research plan. `machine.json` is the lab booking form. You wouldn't submit a lab booking form without knowing what experiment you're running. Same thing here.

Let's trace through this.

---

````{admonition} Real-World Research Example: Graphene + H₂
:class: seealso
The dpgen `param.json` walkthrough below uses a real graphene + H₂ research project. This is a more complex system than our Ar/water tutorial examples. Two elements, multiple system sizes (2 to 448 atoms), energy scale mismatch between isolated H₂ gas and graphene slabs. It shows what a production dpgen run actually looks like, problems and all.
````

## 1. Type Map and Mass Map

```json
"type_map": ["C", "H"],
"mass_map": [12.011, 1.008],
```

Two fields. Four values. Looks trivial.

It is not trivial. Not even close. `type_map` is the single most important consistency requirement in your entire pipeline. And I mean *entire*. Training data conversion. Model architecture. LAMMPS exploration. DFT labeling. Pseudopotential assignment. Everything downstream reads from this ordering. Everything.

Position 0 = Carbon. Position 1 = Hydrogen. That's the contract. Once you write it, every other file in the workflow has to honor it.

Here's what nobody tells you. Nothing enforces this contract. There is no validator. There is no warning. You can flip the ordering in one file and every other file will happily proceed with the wrong assignment. The loss curve looks fine. The forces look reasonable. The model is garbage.

```{admonition} Common Mistake
:class: caution
`type_map` ordering must match **everywhere**:
- Your `type.raw` file (0 = first element, 1 = second, etc.)
- Your POSCAR/structure files (atoms listed in this order)
- Your pseudopotential files in `fp_pp_files` (same order)
- For VASP: your POTCAR concatenation (same order)
- Your `default_training_param.model.type_map` (same list)
- Your `sel` list in the descriptor (same order)

Get any one of these wrong and the model trains on nonsense. <mark class="silent-fail">Nothing crashes. Nothing warns you. The loss curve looks fine. The forces look reasonable. The model silently assigns carbon's pseudopotential to hydrogen and hydrogen's to carbon. Every energy, every force, wrong.</mark> I've seen this waste weeks of compute. Three days on 128 cores. Type_map was backwards. Every force assigned to the wrong element. Nothing crashed. Everything just quietly produced garbage.

Not optional. Not a suggestion. Check it three times. Then check it again.
```

`mass_map` is the atomic mass for each type, used for LAMMPS MD during exploration. Match the order to `type_map`. Standard atomic masses. Nothing fancy. Nobody has ever gotten `mass_map` wrong. `type_map`, on the other hand, will get you eventually. It gets everyone eventually.

That's the foundation. Two arrays. Get them right or nothing else matters.

---

## 2. Initial Training Data

```json
"init_data_prefix": "../init_data",
"init_data_sys": [
    "set_2atoms",
    "set_4atoms",
    "set_8atoms",
    "set_16atoms",
    "set_72atoms",
    "set_74atoms",
    "set_80atoms",
    "set_88atoms",
    "set_96atoms",
    "set_gap_128atoms",
    "set_gap_288atoms",
    "set_gap_2atoms",
    "set_gap_320atoms",
    "set_gap_352atoms",
    "set_gap_384atoms",
    "set_gap_448atoms"
],
"init_batch_size": ["auto","auto","auto","auto","auto","auto","auto","auto","auto","auto","auto","auto","auto","auto","auto","auto"],
```

This is the seed data. The curriculum the model sees before dpgen even starts its loop. First day of class.

And the quality of that first day determines whether the next three weeks go smoothly or spiral into chaos. I've seen both. With bad initial data, the first few iterations produce models that are basically rolling dice. They explore poorly, generate sky-high deviations, and every "candidate" structure is junk. The DFT labels come back, the model retrains on a mix of garbage and noise, and three iterations later you've spent 500 CPU-hours and learned nothing. Three iterations wasted. I learned this the expensive way.

`init_data_prefix` is the path prefix. Each entry in `init_data_sys` is a directory containing DeePMD-format training data: the `set.*/` subdirectories with `box.npy`, `coord.npy`, `energy.npy`, `force.npy`.

In our case, 16 datasets of varying system sizes. 2 atoms = isolated H₂, up to 448 atoms = large graphene slabs with hydrogen. The `set_gap_*` entries are from gap-filling (covered in Practical Topics), manually curated data to fill holes the dpgen loop couldn't reach on its own.

Fair warning: including both isolated H₂ (~-16 eV/atom) and graphene slabs (~-278 eV/atom) in one training set creates an energy scale mismatch. We hit this problem. See [Energy Scale Traps](../practical/energy-scale.md) for how we dealt with it using `atom_ener` reference corrections.

```{admonition} Key Insight
:class: tip
**How much initial data do you need?** Here's the honest answer.

Minimum: 50-100 diverse frames from a short AIMD run or `dpgen init_bulk/init_surf`. That gives the models enough to not be completely random on iteration 0. Enough to stumble forward instead of stumble in circles.

Better: 200+ frames spanning different configurations, temperatures, and system sizes. Now the models have a real foundation. The first exploration produces actual candidates instead of a wall of "failed" frames.

Our real case has ~1200 frames across 16 datasets because we combined AIMD data with gap-filled data from prior failed attempts. Overkill for a first run? Maybe. But those first models were *stable*. They explored without blowing up. That's worth a lot.

Starting with 10 frames? The first iteration's models are basically guessing. The exploration produces thousands of "failed" frames. Nothing gets labeled productively. You're spinning your wheels. I've seen this go wrong too many times.
```

`init_batch_size` controls how many frames per training batch from each dataset. `"auto"` lets DeePMD figure it out based on dataset size. Leave it. I have never had a reason to change this. Not once.

---

## 3. System Configurations

```json
"sys_configs_prefix": "..",
"sys_configs": [
    ["sys_configs/sys_bare/POSCAR"],
    ["sys_configs/sys_4h2/POSCAR"],
    ["sys_configs/sys_8h2/POSCAR"],
    ["sys_configs/sys_12h2/POSCAR"],
    ["sys_configs/sys_h2gas/POSCAR"]
],
"sys_batch_size": ["auto", "auto", "auto", "auto", "auto"],
```

These are the starting structures for LAMMPS exploration in Stage 2. Field trip destinations. Each entry is a different type of structure the model needs to learn about. Different environments, different challenges.

In our case:
- Index 0: Bare graphene (no hydrogen)
- Index 1: Graphene + 4 H₂ molecules
- Index 2: Graphene + 8 H₂ molecules
- Index 3: Graphene + 12 H₂ molecules
- Index 4: Isolated H₂ gas (no graphene)

Yes, Index 4 is isolated H₂ gas alongside graphene slabs. And yes, this creates the energy scale problem we flag in section 5d and in [Energy Scale Traps](../practical/energy-scale.md). We include it because the model needs to handle H₂ desorption (molecules leaving the surface into vacuum). If you skip it, the model has never seen free H₂ and will produce garbage forces the moment a molecule desorbs during exploration. The fix isn't removing it; it's using `atom_ener` reference corrections so the loss function sees comparable residual energies instead of raw -16 vs -278 eV/atom.

The `sys_idx` values in `model_devi_jobs` (coming up soon) reference these indices. When dpgen sees `"sys_idx": [0, 1, 4]`, it reads that as "explore bare graphene, graphene + 4 H₂, and H₂ gas this iteration." Those index numbers are the bridge between your structures and your exploration schedule.

Pay attention to this next part. This one will bite you.

````{admonition} Common Mistake
:class: caution
**`sys_configs` is a 2D list.** Each inner list is one system. Here's what wrong looks like:

```json
// WRONG: flat list
"sys_configs": ["sys_bare/POSCAR", "sys_4h2/POSCAR"]

// RIGHT: list of lists
"sys_configs": [["sys_bare/POSCAR"], ["sys_4h2/POSCAR"]]
```

Miss those inner brackets and dpgen treats each *character* of the path as a separate system. Or it crashes. Or it does something equally unhinged. The error messages won't point you here. You'll stare at the traceback for an hour before you spot the missing brackets. I'm serious.

Each inner list can contain multiple POSCARs for the same system (e.g., different supercell sizes). dpgen picks from the list for that system index. One POSCAR per inner list is the common case.
````

So that's three sections down: element identity, seed data, exploration structures. The skeleton of the run. Now we get to the part that actually does the learning.

---

## 4. Number of Models

```json
"numb_models": 4,
```

One number. It controls the entire active learning mechanism.

Four models. Trained on the same data. Same architecture, same loss function, different random seeds. Four students, same textbook, different study habits.

Why four? Because disagreement is the signal. Throw a new structure at all four models and ask: what's the force on atom 7? If they all say roughly 0.5 eV/Angstrom, that structure is in familiar territory. The model knows this. But if model 1 says 0.5, model 2 says 0.3, model 3 says 0.8, model 4 says 0.4? They're arguing. And that argument is the most valuable data you'll generate in this entire workflow. Those structures, the ones where trained models can't agree, are exactly where your model needs to learn next.

That's the whole trick.

The DP-GEN paper tested this. Four gives a good balance between reliability of the deviation signal and compute cost. And I know what you're thinking. "Four models? That's 4x the GPU time just for training." Yeah, it is. And it's worth it. Because without those four models, you have no uncertainty estimate. You're flying blind. You're adding DFT data randomly instead of strategically. Four models trained in parallel on one GPU takes maybe 30-60 minutes each. The DFT calculations those models save you? Hundreds of CPU-hours.

Could you use 3? If GPU time is really tight. I wouldn't go below 3. With only 2 models, one disagreement could be noise. With 4, the signal is clear.

---

## 5. Training Parameters

Alright. This is the big one.

The `default_training_param` block is the DeePMD-kit input that gets stamped onto every model in every iteration. If you read [Ch 2](../ch02/deepmd-architecture.md), you already know what each piece does conceptually (descriptor, fitting network, loss function, learning rate). Here we focus on the actual values in param.json and why they're set the way they are for our graphene + H₂ system.

```json
"default_training_param": {
    "model": {
        "type_map": ["C", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.0,
            "rcut_smth": 2.0,
            "sel": [60, 120],
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "resnet_dt": false
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": true
        }
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 0.001,
        "stop_lr": 5e-08,
        "decay_steps": 5000
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 2.0,
        "start_pref_f": 1000,
        "limit_pref_f": 1.0,
        "start_pref_v": 0.0,
        "limit_pref_v": 0.0
    },
    "training": {
        "training_data": {
            "systems": [],
            "batch_size": "auto"
        },
        "numb_steps": 1000000,
        "seed": 42,
        "disp_file": "lcurve.out",
        "disp_freq": 1000,
        "save_freq": 10000,
        "save_ckpt": "model.ckpt"
    }
}
```

That is a wall of JSON. Let me take it apart piece by piece.

### 5a. The Descriptor

The descriptor is how the model sees its neighborhood. [Ch 2](../ch02/deepmd-architecture.md#the-descriptor-how-atoms-see-their-neighbors) covers the full picture (smoothing, invariance properties, embedding network). Here's what matters for your param.json.

```{admonition} Config Walkthrough
:class: note
- **`type: "se_e2_a"`**: The workhorse descriptor. DeePMD v3 has newer options (`se_atten`, `dpa2`) but `se_e2_a` is battle-tested. Start here.

- **`rcut: 6.0`**: Cutoff radius in Angstroms. For covalent systems (C-C ~1.4 Angstroms, C-H ~1.1 Angstroms), 6 Angstroms captures several neighbor shells. For metals or ionic systems, bump to 8-9. Don't go past 9 or 10 without a good reason. Neighbor count scales as $r^3$, so going from 6 to 9 roughly *triples* your compute.

- **`rcut_smth: 2.0`**: Smoothing width. Neighbor contributions taper to zero between 4.0 and 6.0 Angstroms. Leave this at 2.0. I have never had a reason to change it.

- **`sel: [60, 120]`**: Maximum neighbors per element type within `rcut`. `sel[0] = 60` → up to 60 Carbon neighbors. `sel[1] = 120` → up to 120 Hydrogen neighbors. <mark class="hard-req">Order matches `type_map`. Not optional.</mark> If your densest configuration has 70 carbon neighbors and you set `sel[0] = 60`, <mark class="silent-fail">DeePMD silently ignores the 10 farthest carbons. No warning. No error. Just subtly wrong descriptors.</mark>

  Why 60 and 120? In our densest expected configuration, an atom sees roughly 50 carbons and 80 hydrogens within 6 Angstroms. We pad by ~20%. Memory is cheap. Silent truncation is not.

- **`neuron: [25, 50, 100]`**: Embedding network layers. Standard size. Smaller systems (CH₄, water) can get away with `[10, 20, 40]`.

- **`axis_neuron: 16`**: Angular encoding dimensionality. 16 is the default. Rarely needs changing.

- **`resnet_dt: false`**: Standard for `se_e2_a`. Don't touch this.
```

### 5b. The Fitting Network

The fitting network takes the descriptor vector and outputs one number: this atom's energy contribution. See [Ch 2](../ch02/deepmd-architecture.md#the-fitting-network-from-environment-to-energy) for the full explanation. Here's the sizing decision.

```json
"fitting_net": {
    "neuron": [240, 240, 240],
    "resnet_dt": true
}
```

Why `[240, 240, 240]`? Because our system has physisorption wells, surface diffusion barriers, interlayer interactions, and hydrogen-hydrogen repulsion all happening simultaneously. I tried `[100, 100, 100]` first. The forces on the hydrogen atoms never got below 80 meV/Angstrom. Bumped to `[240, 240, 240]`. Dropped to 30.

| System complexity | Typical fitting net |
|---|---|
| Simple (single element, bulk metal) | `[60, 60, 60]` |
| Medium (small molecules, water) | `[100, 100, 100]` |
| Complex (interfaces, multi-element) | `[240, 240, 240]` |

`resnet_dt: true` enables residual connections. Leave it on. Always.

### 5c. Learning Rate

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 5e-08,
    "decay_steps": 5000
}
```

Exponential decay over 1M steps, dropping ~4 orders of magnitude. [Ch 2](../ch02/deepmd-architecture.md#the-learning-rate-how-fast-the-model-learns) explains the math. For param.json purposes: just use these values. The only time I've needed to change them is when training diverged (NaN in the loss). If that happens, halve `start_lr` to 0.0005. That has fixed it every single time.

### 5d. Loss Function

```json
"loss": {
    "type": "ener",
    "start_pref_e": 0.02,
    "limit_pref_e": 2.0,
    "start_pref_f": 1000,
    "limit_pref_f": 1.0,
    "start_pref_v": 0.0,
    "limit_pref_v": 0.0
}
```

The loss prefactors shift during training. Forces dominate early (50,000:1 over energy), then energy catches up to roughly 2:1. [Ch 2](../ch02/deepmd-architecture.md#the-loss-function-what-the-model-optimizes) covers *why* this works (information density in forces, shape-before-elevation). Here's what you need to decide for your system:

- **Force vs energy weighting**: These defaults work for most systems. Don't change them unless you have a specific reason.
- **Virial prefactors**: Zeroed out here because our QE calculations don't produce reliable stress tensors for 2D slab systems. For bulk 3D systems where you need accurate pressure, set `start_pref_v = 0.02`, `limit_pref_v = 1.0`.

```{admonition} Warning: Energy Scale
:class: danger
Our config includes both isolated H₂ (~-16 eV/atom) and graphene slabs (~-278 eV/atom). Without correction, the loss function focuses on reducing error on the slabs because those energies are simply larger numbers. The H₂ forces could be terrible and the total loss barely flinches.

<mark class="hard-req">If you mix systems with very different energy scales, you need `atom_ener` reference corrections.</mark> Without them the model silently sacrifices accuracy on the minority system. See [Energy Scale Traps](../practical/energy-scale.md) for the full story and our solution.
```

### 5e. Training Settings

```json
"training": {
    "training_data": {
        "systems": [],
        "batch_size": "auto"
    },
    "numb_steps": 1000000,
    "seed": 42,
    "disp_file": "lcurve.out",
    "disp_freq": 1000,
    "save_freq": 10000,
    "save_ckpt": "model.ckpt"
}
```

Fast section. These are mostly set-and-forget.

- **`systems: []`**: Leave it empty. dpgen fills this automatically with the current training data at each iteration. If you put paths here, dpgen ignores them anyway.
- **`numb_steps: 1000000`**: One million training steps for fresh models. Takes ~30-60 minutes on a decent GPU. For simple systems, 400k is enough. For complex multi-element systems, 1M is the safe bet.
- **`seed: 42`**: Doesn't matter what you put here. dpgen overrides this with different random seeds for each of the 4 models. That's the whole point: same architecture, same data, different initialization. The value 42 is tradition, not science.
- **`disp_file: "lcurve.out"`**: Where the learning curve gets written. This is the file you'll stare at when something feels off. Columns: step, learning rate, energy RMSE (train), energy RMSE (val), force RMSE (train), force RMSE (val).
- **`disp_freq: 1000`**: Write to `lcurve.out` every 1000 steps. Fine-grained enough to spot problems, coarse enough to not bloat the file.
- **`save_freq: 10000`**: Checkpoint every 10k steps. If training crashes at step 990,000, you lose at most 10k steps instead of the whole run. Set this lower if your jobs are unstable. The checkpoints take almost no space.

Alright. That was the entire training block. The model architecture, the learning rate, the loss function, the training loop. All of it lives inside `default_training_param`. dpgen stamps this onto every model in every iteration automatically. You write it once. It runs everywhere.

Clean.

---

## 6. Training Reuse (Transfer Learning)

```json
"training_reuse_iter": 5,
"training_reuse_numb_steps": 400000,
"training_reuse_start_lr": 0.0001,
"training_reuse_start_pref_e": 1.0,
"training_reuse_start_pref_f": 100,
"training_reuse_old_ratio": "auto",
```

For the first 5 iterations, dpgen trains every model from scratch. One million steps. Random initialization. Full training.

Starting from iteration 5, it switches to **transfer learning**: take the model from the previous iteration and fine-tune it with the new data. The model already knows most of what it needs. You're just teaching it the new stuff. Like a student who's passed the midterm; you don't make them retake the intro lectures.

Why iteration 5 and not 0? Because the first few iterations add a lot of diverse data. The model's internal representation might need to reorganize significantly. Fine-tuning a bad model on new data locks in bad patterns. After iteration 5, the model is stable enough that building on top of it makes sense. Before that, you're better off starting fresh.

- **`training_reuse_numb_steps: 400000`**: Only 400k steps instead of 1M. Fine-tuning converges faster because you're starting from a good place instead of random weights.
- **`training_reuse_start_lr: 0.0001`**: Ten times smaller than fresh training. You don't want to overwrite what the model already knows. Small adjustments, not a renovation.
- **`training_reuse_start_pref_e/f`**: More balanced loss weights. `pref_f = 100` instead of 1000. The model's forces are already decent; you don't need to hammer them as hard.
- **`training_reuse_old_ratio: "auto"`**: How much old vs. new data to use. `"auto"` handles it. Let it.

Over a 15-iteration run, transfer learning cuts training time roughly in half from iteration 5 onward. On 4 models, that adds up fast. Real time saved on real GPUs.

---

## 7. Exploration Settings

```json
"model_devi_dt": 0.0005,
"model_devi_skip": 0,
"model_devi_f_trust_lo": 0.05,
"model_devi_f_trust_hi": 0.15,
"model_devi_clean_traj": 3,
```

This is where the active learning happens. Five fields. Two of them control the entire intelligence of the dpgen loop. I'm going to slow down here because this is the part the docs skip over too quickly.

```{admonition} Config Walkthrough
:class: note
- **`model_devi_dt: 0.0005`**: LAMMPS timestep in picoseconds. That's 0.5 fs. Conservative, and intentionally so. We have hydrogen in the system. Light atoms vibrate fast. A 2 fs timestep would work for bulk copper, but for anything with hydrogen, 0.5 fs keeps the dynamics stable. If your system is all heavy atoms (no H, no Li), 1-2 fs is fine.

- **`model_devi_skip: 0`**: Number of frames to skip at the beginning of each trajectory before analyzing deviations. Zero means analyze everything from the first frame. Set this to 1000 or so if you want to discard the equilibration period and only analyze the production run.

- **`model_devi_f_trust_lo: 0.05`**: The "I've got this" threshold. Force deviation below 0.05 eV/Angstrom across all four models means they agree. The structure is well-learned territory. No need to waste DFT compute on it.

- **`model_devi_f_trust_hi: 0.15`**: The "this is nonsense" threshold. Force deviation above 0.15 eV/Angstrom means the models are so confused that the structure is probably unphysical or completely outside the training distribution. Labeling it won't help. Like a student so lost that answering one more question won't fix the underlying gap.

- **`model_devi_clean_traj: 3`**: Keep trajectories from the last 3 iterations only. Older ones get deleted. These LAMMPS dump files are huge. A single trajectory can be hundreds of MB. Over 15 iterations with 30 simulations each, you're looking at terabytes. Clean them up.
```

<mark class="key-insight">The magic is in the middle. Structures with force deviations between 0.05 and 0.15 eV/Angstrom are the **candidates**.</mark> The models are uncertain but not completely lost. Hard but learnable. These are the structures that get sent to DFT. And these, specifically these, are what the model actually learns from.

Two numbers. `trust_lo` and `trust_hi`. That's the entire active learning signal.

Are you seeing this? The whole elaborate dpgen machinery, the four models, the LAMMPS exploration, the DFT labeling, all of it boils down to sorting structures into three buckets based on two thresholds.

| Bucket | Condition | Action |
|--------|-----------|--------|
| Accurate | deviation < `trust_lo` | Skip. Model knows this. |
| Candidate | `trust_lo` ≤ deviation ≤ `trust_hi` | Send to DFT. Model needs this. |
| Failed | deviation > `trust_hi` | Skip. Model is too confused. |

Simple. Effective. And very easy to get wrong.

```{admonition} Common Mistake
:class: caution
I set `trust_lo` to 0.25 on my first run. Do you know what happened? Every single structure was classified "accurate." The model stopped getting new training data. For three iterations, it trained on the same dataset, explored the same territory, found zero candidates, and I sat there wondering why it wasn't converging.

Three iterations of zero progress. On an HPC cluster that charges by the hour.

It was grade inflation. The threshold was so loose that even genuinely uncertain predictions passed. The model sat there, smug, convinced it knew everything. It didn't. It was just that my bar was so low even wildly uncertain predictions cleared it.

Then I changed it to 0.05. Candidates everywhere. The model actually started learning.

Start tight: 0.05 for `trust_lo`, 0.15 for `trust_hi`. Only loosen if you're drowning in candidates (hundreds per iteration). If you're getting zero candidates and the model clearly hasn't converged, check `trust_lo` first. Trust me on this one.
```

### 7a. The model_devi_jobs Schedule

This is the exploration schedule. One entry per dpgen iteration. Each entry tells dpgen which structures to explore, at what temperatures, for how long. This is where you design the model's education, semester by semester.

```json
"model_devi_jobs": [
    {
        "_comment": "Iter 0: Light exploration, bare + 4H2 + gas",
        "sys_idx": [0, 1, 4],
        "temps": [77, 150, 300],
        "nsteps": 50000,
        "ensemble": "nvt",
        "trj_freq": 100
    },
    {
        "_comment": "Iter 1: Add 8H2",
        "sys_idx": [0, 1, 2, 4],
        "temps": [77, 300, 500],
        "nsteps": 200000,
        "ensemble": "nvt",
        "trj_freq": 100
    },
    {
        "_comment": "Iter 2: Full set including 12H2",
        "sys_idx": [0, 1, 2, 3, 4],
        "temps": [77, 300, 500],
        "nsteps": 500000,
        "ensemble": "nvt",
        "trj_freq": 1000
    },
    {
        "_comment": "Iter 3: Long exploration, wider T range",
        "sys_idx": [0, 1, 2, 3, 4],
        "temps": [50, 77, 200, 300, 500, 800],
        "nsteps": 1000000,
        "ensemble": "nvt",
        "trj_freq": 1000
    },
    {
        "_comment": "Iter 4: Convergence check",
        "sys_idx": [0, 1, 2, 3, 4],
        "temps": [50, 77, 200, 300, 500, 800],
        "nsteps": 2000000,
        "ensemble": "nvt",
        "trj_freq": 1000
    }
]
```

Look at the progression. Each iteration pushes harder. The model earns its way to more difficult conditions.

| Iteration | Systems | Temps (K) | Steps | Frames/traj |
|-----------|---------|-----------|-------|-------------|
| 0 | 3 | 77, 150, 300 | 50k | 500 |
| 1 | 4 | 77, 300, 500 | 200k | 2000 |
| 2 | 5 | 77, 300, 500 | 500k | 500 |
| 3 | 5 | 50-800 | 1M | 1000 |
| 4 | 5 | 50-800 | 2M | 2000 |

Iteration 0 is gentle. Three systems, mild temperatures, short run. You don't throw a freshly trained model into 800 K on day one. It would blow up in 200 steps, produce nothing but "failed" frames, and you'd have burned GPU time on exploration that generated zero usable candidates.

By iteration 3, the model has seen hundreds of DFT-labeled structures across multiple configurations and temperatures. Now you push it: 50 K to 800 K, 1 million steps, all five systems. This is where you find out if your model actually learned the physics or just memorized the specific configurations it was trained on.

Iteration 4 is the final exam. Two million steps. Everything. If it passes with >98% accurate frames, you're done.

dpgen creates one LAMMPS task for **every combination** of `sys_idx` and temperature. Iteration 0: 3 systems times 3 temperatures = 9 simulations. Iteration 3: 5 times 6 = 30 simulations. Each generates `nsteps / trj_freq` frames for deviation analysis.

Check the numbers. Iteration 3 produces 30 trajectories, each with 1000 frames. That's 30,000 frames that need deviation analysis. From those 30,000, maybe 50-200 end up as candidates for DFT. The funnel is aggressive by design.

```{admonition} Key Insight
:class: tip
**Design your exploration schedule with intent. Here are the principles I learned the hard way:**

1. **Start small.** Few systems, mild temperatures, short runs. The initial model is rough. Long explorations at this stage just produce thousands of "failed" frames that get thrown away. Wasted GPU time.

2. **Add complexity gradually.** New systems in iteration 1 or 2, not iteration 0. Higher temperatures only after the model has learned the basics. You're building on a foundation. If the foundation is shaky, adding load collapses it.

3. **Increase `trj_freq` for long runs.** 50k steps at `trj_freq=100` gives 500 frames. Fine. But 2M steps at `trj_freq=100` gives 20,000 frames per trajectory. That's absurd. Most are redundant because consecutive MD frames are highly correlated. Use `trj_freq=1000` for anything over 200k steps. I cannot stress this enough.

4. **The last iteration should be the hardest test you can design.** Widest temperature range, longest runs, all systems. If the model survives this with >98% accurate, it's ready for production. That's your graduation day.
```

```{admonition} HPC Reality
:class: warning
Total LAMMPS simulations per iteration = `len(sys_idx) * len(temps)`. With 5 systems and 6 temperatures, that's 30 GPU jobs. Each runs for 2M steps at 0.5 fs with a deep potential. On a single GPU, each takes maybe 2-6 hours depending on system size.

That's 60-180 GPU-hours per iteration. Just for exploration. And you haven't even started the DFT labeling yet.

Make sure your `machine.json` `group_size` for `model_devi` is set to pack multiple simulations per GPU job where possible. Otherwise you're submitting 30 separate PBS jobs per iteration, and the scheduler overhead alone will eat your afternoon. Queue wait time times 30 is not a number you want to look at.
```

---

## 8. First-Principles Settings

```json
"fp_style": "pwscf",
"fp_task_max": 50,
"fp_task_min": 5,
"fp_accurate_threshold": 0.98,
"fp_accurate_soft_threshold": 0.9,
```

This controls the DFT labeling stage. The teacher grading the hard questions. The whole point of active learning is that the teacher only grades the questions the student found difficult. Not the ones they already know. Not the ones that are beyond comprehension. Just the ones in the sweet spot.

```{admonition} Config Walkthrough
:class: note
- **`fp_style: "pwscf"`**: Use Quantum ESPRESSO as the DFT engine. The other main option is `"vasp"`. This determines how dpgen generates input files and parses output. Pick the one installed on your HPC. We use QE.

- **`fp_task_max: 50`**: Maximum number of candidate structures sent to DFT per iteration. Even if 200 structures are candidates, only 50 get labeled. They're selected randomly from the candidate pool. This is your DFT budget cap. Think of it as grading capacity. The teacher can only grade so many exams per week. 50 structures at ~1 CPU-hour each = 50 CPU-hours per iteration. Plan this around your HPC allocation, not around what's theoretically optimal.

- **`fp_task_min: 5`**: Minimum DFT tasks per iteration. Even if only 2 structures are candidates, dpgen samples at least 5 (pulling in some "accurate" structures) to ensure meaningful data addition. Prevents empty or near-empty iterations where you burn a full training cycle for 2 new frames.

- **`fp_accurate_threshold: 0.98`**: The convergence criterion. If 98% of explored frames are "accurate" (below `trust_lo`), dpgen considers this system learned. The model knows its stuff. This is the finish line.

- **`fp_accurate_soft_threshold: 0.9`**: Above 90% accurate, dpgen starts reducing the number of candidates it sends. A soft transition. The model is getting close, so dpgen eases off the DFT throttle gradually rather than going from full speed to zero. Elegant engineering.
```

### 8a. Pseudopotentials

```json
"fp_pp_path": "pseudo",
"fp_pp_files": [
    "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H.pbe-rrkjus_psl.1.0.0.UPF"
],
```

**The order of `fp_pp_files` must match `type_map`.** Carbon first, hydrogen second. Yes, I'm repeating myself from Section 1. No, I will not apologize. This is the second place where element ordering bites people, and the consequences are identical: silent garbage.

`fp_pp_path` is relative to the dpgen working directory. Put your `.UPF` files in a `pseudo/` subdirectory next to your `param.json`. dpgen copies these into each DFT task directory automatically.

For VASP users: you don't list individual pseudopotential files. Instead, dpgen constructs the POTCAR by concatenating per-element POTCARs from `fp_pp_path` in `type_map` order. But the **ordering still must match**. This is not a QE-specific rule. This is a universal rule.

### 8b. QE Input Parameters

Now we slow down. This section determines the quality of every DFT label your model trains on. Get one setting wrong and every label is subtly off. The model trains on those labels. The model inherits those errors. And nothing tells you it happened.

```json
"user_fp_params": {
    "control": {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "outdir": "./OUT",
        "tprnfor": true,
        "tstress": true,
        "disk_io": "none"
    },
    "system": {
        "ecutwfc": 50,
        "ecutrho": 400,
        "input_dft": "PBE",
        "vdw_corr": "dft-d3",
        "dftd3_version": 4,
        "nosym": true,
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.005
    },
    "electrons": {
        "conv_thr": 1e-06,
        "electron_maxstep": 200,
        "mixing_beta": 0.3
    },
    "kspacing": 0.5
}
```

This is the QE input template dpgen stamps onto every DFT calculation in the fp stage. Every single one. <mark class="hard-req">Consistency is everything here. Every DFT calculation across all dpgen iterations must use identical settings.</mark> Changing the functional mid-run is like changing the grading rubric halfway through the semester. The model sees conflicting labels and can't learn.

````{admonition} Config Walkthrough
:class: note
**`control` section:**
- `calculation: "scf"`: Single-point energy calculation. dpgen only needs energies and forces, not geometry relaxation. Never set this to "relax" or "vc-relax" in a dpgen context. The candidate structures come from MD; you want to label them as-is.
- **`tprnfor: true`**: Print forces. **Not optional. Not a suggestion.** Without this flag, QE computes forces internally but doesn't write them to the output file. <mark class="silent-fail">dpdata reads the output, finds no forces, and your training data gets zero-force labels.</mark> The loss looks fine. The model trains. The forces are all zeros. Garbage in, garbage out. I lost two days to this. QE doesn't complain. It just silently gives you garbage. Helpful.
- `tstress: true`: Print the stress tensor. Useful if you're training with virial prefactors. Costs almost nothing to include.
- `disk_io: "none"`: Don't write wavefunctions to disk. We don't need them and they waste I/O on the cluster.

**`system` section:**
- `ecutwfc: 50`: Plane-wave cutoff in Ry. 50 Ry is standard for PAW pseudopotentials. This should match whatever you used for your initial AIMD data. Don't change it between runs. Consistency.
- `ecutrho: 400`: Charge density cutoff. 8x `ecutwfc` for PAW. For ultrasoft pseudopotentials, use 10-12x.
- `nosym: true`: Disable symmetry detection. Important. The candidate structures come from MD trajectories and have no symmetry. QE trying to find symmetry wastes time and occasionally causes convergence issues on slightly distorted structures.
- `occupations/smearing/degauss`: Gaussian smearing at 0.005 Ry. Standard for metallic or semi-metallic systems. Graphene is a semimetal, so we need this.

**`electrons` section:**
- `conv_thr: 1e-06`: SCF convergence threshold in Ry. Tight enough for reliable forces. Don't go looser than 1e-5. Loose convergence means noisy forces, which means noisy training labels, which means the model learns noise. It doesn't crash. It just doesn't learn well. You'll see it as a plateau in the force RMSE that never breaks below 40-50 meV/Angstrom no matter how many iterations you run.
- `electron_maxstep: 200`: Maximum SCF iterations. 200 is generous. If a calculation needs more than 200 steps to converge, something is wrong with the structure. Let it fail. dpgen handles failures gracefully.
- `mixing_beta: 0.3`: Charge mixing parameter. Lower = more stable convergence but slower. If your DFT jobs fail to converge, try 0.1 or 0.2 before anything else.

**`kspacing: 0.5`**: K-point spacing in reciprocal Angstroms. dpgen generates the k-point grid automatically from this value and the cell dimensions. 0.5 is reasonable for a slab system. For bulk metals, use 0.3 or tighter.
````

```{admonition} Common Mistake
:class: caution
**`input_dft` for vdW functionals on QE 7.3.1**: If you need a vdW-DF functional, use `'vdw-df2-b86r'`, NOT `'rev-vdw-df2'`. The latter is an obsolete label that older QE versions accepted. QE 7.3.1 on most HPCs won't recognize it and will either crash or silently fall back to a different functional.

Silently. Fall back. To a different functional.

Let that sink in. Your DFT calculations run successfully, produce energies and forces, and dpgen happily adds them to the training set. Except they were computed with the wrong functional. Every label is subtly wrong. The model trains. It looks converged. The forces are off by just enough to ruin your production simulations. You won't catch it until validation, weeks later.

In our config, we're using `"PBE"` with explicit D3(BJ) correction (`vdw_corr: "dft-d3"`, `dftd3_version: 4`) instead of a vdW-DF functional. This is a valid alternative for van der Waals interactions. The key point: be consistent. Same functional. Same pseudopotentials. Same cutoffs. Every iteration. No exceptions.
```

---

## 9. For VASP Users

If you're using VASP instead of QE, the `fp_style` and related fields look different:

```json
"fp_style": "vasp",
"user_incar_params": {
    "ENCUT": 400,
    "EDIFF": 1e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "Auto",
    "ISYM": 0,
    "PREC": "Accurate",
    "LWAVE": false,
    "LCHARG": false,
    "NSW": 0,
    "IBRION": -1,
    "KSPACING": 0.3
}
```

The key mappings between QE and VASP:

| Purpose | QE | VASP |
|---------|-----|------|
| Disable symmetry | `nosym: true` | `ISYM: 0` |
| Single-point (no relax) | `calculation: "scf"` | `NSW: 0`, `IBRION: -1` |
| Don't dump wavefunctions | `disk_io: "none"` | `LWAVE: false`, `LCHARG: false` |
| Plane-wave cutoff | `ecutwfc` (Ry) | `ENCUT` (eV) |
| SCF convergence | `conv_thr` | `EDIFF` |
| K-point spacing | `kspacing` | `KSPACING` |

One difference: VASP always writes forces. No `tprnfor` equivalent to forget. Small mercy.

Pseudopotential handling: VASP reads from a POTCAR that dpgen constructs by concatenating per-element POTCARs in `type_map` order from `fp_pp_path`. Same ordering rule applies. Always.

---

## The Complete param.json

Here's everything together. The real config from the graphene + H₂ project, comments stripped:

```json
{
    "type_map": ["C", "H"],
    "mass_map": [12.011, 1.008],
    "init_data_prefix": "../init_data",
    "init_data_sys": [
        "set_2atoms", "set_4atoms", "set_8atoms", "set_16atoms",
        "set_72atoms", "set_74atoms", "set_80atoms", "set_88atoms",
        "set_96atoms", "set_gap_128atoms", "set_gap_288atoms",
        "set_gap_2atoms", "set_gap_320atoms", "set_gap_352atoms",
        "set_gap_384atoms", "set_gap_448atoms"
    ],
    "sys_configs_prefix": "..",
    "sys_configs": [
        ["sys_configs/sys_bare/POSCAR"],
        ["sys_configs/sys_4h2/POSCAR"],
        ["sys_configs/sys_8h2/POSCAR"],
        ["sys_configs/sys_12h2/POSCAR"],
        ["sys_configs/sys_h2gas/POSCAR"]
    ],
    "numb_models": 4,
    "default_training_param": { "...training config..." },
    "model_devi_dt": 0.0005,
    "model_devi_f_trust_lo": 0.05,
    "model_devi_f_trust_hi": 0.15,
    "model_devi_jobs": [ "...exploration schedule..." ],
    "fp_style": "pwscf",
    "fp_task_max": 50,
    "fp_task_min": 5,
    "user_fp_params": { "...QE settings..." },
    "fp_pp_path": "pseudo",
    "fp_pp_files": ["C.pbe-n-kjpaw_psl.1.0.0.UPF", "H.pbe-rrkjus_psl.1.0.0.UPF"]
}
```

That's it. One file. Every scientific decision.

## Writing Your Own: A Checklist

Before you launch `dpgen run`, go through this list. All of it. Every time. I'm serious.

1. **`type_map`** matches your structures, pseudopotentials, and training config? **Check three times.** Then check it a fourth time. I'm not being dramatic. This is the number one source of silent failures.
2. **`init_data_sys`** points to real directories with valid DeePMD data? Run `ls init_data/set_*/set.000/` to verify. If any directory is missing `energy.npy` or `force.npy`, you'll get a cryptic error at training time.
3. **`sys_configs`** is a 2D list (list of lists) and each POSCAR actually exists at the specified path?
4. **`sel`** in the descriptor is large enough for your densest configuration? Pad by 20%. Memory is cheap. Silent truncation is not.
5. **`model_devi_jobs`** starts gentle and increases in difficulty? First iteration should not have your widest temperature range or longest simulation.
6. **`trust_lo` and `trust_hi`** are reasonable? 0.05 and 0.15 is a safe start. Tighten or loosen based on what you see in the first 2-3 iterations.
7. **`fp_task_max`** is within your HPC budget? 50 DFT calculations at ~1 CPU-hour each = 50 CPU-hours per iteration. Over 10 iterations, that's 500 CPU-hours just for labeling. Know your allocation.
8. **`user_fp_params`** has `tprnfor: true` (QE)? Is it consistent with the DFT settings you used for your initial data?
9. **`fp_pp_files`** order matches `type_map`? Carbon first if `type_map` says carbon first. Always.
10. **`kspacing`** is appropriate for your system size? Too coarse for bulk metals, too fine wastes compute on large supercells.

If all 10 pass, you're ready for Ch 8 (machine.json) and Ch 9 (running it).

## What's Next

`param.json` tells dpgen *what* to do. Now it needs to know *where* to do it. Which GPU for training, which PBS queue for DFT, how many cores per job, how to launch the Apptainer container. That's `machine.json`. Grab your terminal.
