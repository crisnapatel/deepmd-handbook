# init_bulk and init_surf: Generating Initial Training Data

Before dpgen can learn anything, it needs something to learn FROM. You cannot start the active learning loop with zero data. The model would be random noise pretending to be a potential energy surface. LAMMPS would crash on the first timestep. Nothing happens.

So you need initial training data. Think of it as the first day of class. Just enough so the student is not completely lost. Not a full education; just enough to survive week one.

You have a choice: generate it by hand (tedious, full control) or let dpgen do the heavy lifting. But here is the thing. `dpgen init_bulk` and `dpgen init_surf` are purpose-built for specific system types. If your system fits neatly into "bulk crystal" or "surface slab," these tools will save you hours of manual work. If it doesn't fit (like graphene with adsorbed H2), they will just frustrate you, and you are better off doing it yourself.

Let me walk you through both paths.

---

## The Chicken-and-Egg Problem

dpgen's active learning loop (`dpgen run`) needs `init_data_sys` to start. That is initial training data in DeePMD's npy format. But where does that initial data come from?

Three options:

1. **Manual.** Run a few DFT calculations yourself, convert with dpdata. Full control, full effort.
2. **`dpgen init_bulk`.** Automated pipeline for bulk crystal systems.
3. **`dpgen init_surf`.** Automated pipeline for surface slab systems.

Options 2 and 3 automate the boring parts: generating supercells, perturbing atoms, scaling volumes, submitting DFT jobs, collecting results. They output ready-to-use `init_data_sys` directories.

So why not always use them? Because they were designed for textbook systems. Silicon. Copper. Clean FCC surfaces. Your system might not be a textbook system. And forcing a non-textbook system through a textbook tool is how you waste a day wondering why the output doesn't make sense.

---

## `dpgen init_bulk`

For bulk crystal systems. Silicon. Copper. Diamond. Anything where your starting point is a periodic unit cell that repeats in all three directions.

**Input:** POSCAR (crystal structure) + `param.json` (init config) + `machine.json` (HPC settings).

**What it does:** Takes your unit cell, builds a supercell, randomly perturbs atomic positions, scales the cell volume to different compressions and expansions, runs DFT on every configuration, collects results into DeePMD npy format.

That is a lot of steps. All automated. One command.

### Key Parameters

```json
{
    "stages": [1, 2],
    "cell_type": "diamond",
    "super_cell": [2, 2, 2],
    "elements": ["Si"],
    "potcars": ["Si"],
    "cell_pert_frac": 0.03,
    "atom_pert_distance": 0.01,
    "scale": [0.98, 0.99, 1.00, 1.01, 1.02],
    "pert_numb": 30
}
```

Let me unpack what each field actually controls.

- **`super_cell`: [2, 2, 2]**: How many unit cells in each direction. A 2x2x2 supercell of an 8-atom diamond Si cell gives 64 atoms. Bigger supercells give more diverse local environments but cost more in DFT. Start with 2x2x2. You can always go bigger later.

- **`cell_pert_frac`: 0.03**: Random strain up to 3% on each lattice vector. This generates structures that are not sitting at perfect equilibrium. The model needs to learn what happens when the lattice is slightly distorted, not just what equilibrium looks like. Give the crystal a shake.

- **`atom_pert_distance`: 0.01**: Displace each atom up to 0.01 A from its ideal position. A gentle nudge. Enough to break symmetry without producing something unphysical. If every atom sits at its perfect lattice site, the forces are all zero by symmetry. Zero-force training data teaches the model nothing about how atoms respond to displacement, which is the entire point of a force field.

- **`scale`: [0.98, 0.99, 1.00, 1.01, 1.02]**: Volume scaling factors. Compress, equilibrium, expand. 5 scales times 30 perturbations = 150 DFT calculations. That is your initial data budget.

- **`pert_numb`: 30**: Perturbed configurations per scale factor. 30 gives decent statistical coverage. Less than 10 and you are undersampling. More than 50 and you are spending DFT compute that the active learning loop would have spent more wisely.

```{admonition} Key Insight
:class: tip
Cell perturbation + atomic displacement + volume scaling creates structures that sample the PES around equilibrium. The model learns what happens when you compress, expand, and distort the crystal. What it does NOT learn: high-temperature dynamics, melting, defects, or surfaces. Those are what the dpgen active learning loop is for. The init data just needs to keep the model alive long enough for the loop to start.

That is the whole point. The init data is not a curriculum. It is a survival kit.
```

```console
$ dpgen init_bulk param.json machine.json
```

Hit enter. dpgen submits all DFT jobs, waits for them to finish, collects results. Depending on your queue, you go get coffee. Or lunch.

---

## `dpgen init_surf`

Same idea, but for surface slabs. It takes a bulk structure, cleaves it along specified Miller indices, adds vacuum, and generates perturbed configurations.

```json
{
    "cell_type": "fcc",
    "latt": 3.615,
    "super_cell": [2, 2, 1],
    "elements": ["Cu"],
    "z_min": 9,
    "vacuum_max": 9,
    "vacuum_resol": [0.5, 1.0],
    "millers": [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
    "scale": [0.98, 1.00, 1.02],
    "pert_numb": 10
}
```

Let me trace through this.

- **`millers`**: Surface orientations to generate. [1,0,0], [1,1,0], [1,1,1] covers the three low-index faces. That is usually enough to get started. If you need higher-index surfaces, add them, but know that each one multiplies your DFT cost.

- **`z_min`: 9**: Minimum slab thickness in Angstroms. Thick enough that atoms in the middle behave like bulk. If the slab is too thin, both surfaces interact through the slab. The model learns surface physics contaminated by finite-size artifacts.

- **`vacuum_max`: 9**: Vacuum above the slab in Angstroms. Prevents periodic images from interacting across the vacuum gap. 9 A is fine for most systems. If you are worried, test convergence of the surface energy with vacuum thickness. But 9 A is a safe default.

- **`super_cell`: [2, 2, 1]**: Note the 1 in z. You do not replicate vertically. You only expand in-plane. Replicating in z would just double the slab thickness, not create more diverse surface environments. The diversity comes from the different Miller indices and perturbations, not from stacking slabs.

```console
$ dpgen init_surf param.json machine.json
```

Same workflow. Submit, wait, collect.

---

## How Much Initial Data Is "Enough"?

**50-200 frames.** That is the target. Start there.

- **Too few (<30):** The model cannot run stable MD in the first exploration step. LAMMPS crashes, atoms fly apart, dpgen stalls at iteration 0. You are stuck before you started.
- **Too many (>500):** Wasted DFT compute. dpgen's active learning would have selected more informative structures than your brute-force perturbations ever could. You just burned your allocation on data that is mostly redundant. The active learning loop exists precisely because humans are bad at guessing which configurations matter most.
- **50-200:** A rough but functional model. Wrong about many things, but stable enough to run MD without exploding. The loop starts. The model begins to learn for real.

```{admonition} Practical Rule
:class: note
Your initial data needs to answer one question: can the model run MD without crashing? It does not need to be accurate. It does not need to predict correct surface energies or diffusion barriers. It just needs to keep atoms from flying apart during the first exploration step. Accuracy comes from the loop. The init data just buys you entry.
```

---

## The Manual Approach: When init_bulk/init_surf Don't Fit

`init_bulk` and `init_surf` work great for textbook systems. FCC copper. Diamond silicon. Clean low-index surfaces. But what about:

- A 2D material (graphene), which is not bulk and not a traditional surface?
- A slab with adsorbates (graphene + H2), since init_surf does not know how to place molecules?
- An amorphous structure with no unit cell to perturb?

Those tools will look at your system and have no idea what to do with it. Don't force it. Go manual.

**Step 1:** Run AIMD for 100-500 steps. NVT, 300 K, 0.5-1.0 fs time step. For QE: `calculation = 'md'` with `ion_dynamics = 'verlet'` in `&IONS`. Short, cheap, and it gives you decorrelated snapshots of atoms jiggling around at finite temperature. This is a field trip for your atoms: send them somewhere interesting and record what happens.

**Step 2:** Extract snapshots every 5th-10th frame. <mark class="hard-req">Consecutive MD frames are nearly identical</mark> (the atoms barely moved between two 0.5 fs steps). You need decorrelated structures, not 500 copies of basically the same thing. If you skip this step, you end up training on highly correlated data. The model memorizes one trajectory instead of learning general physics.

**Step 3:** Convert with dpdata.

```python
import dpdata

data = dpdata.LabeledSystem('output/', fmt='qe/pw/scf')
print(f"Loaded {len(data)} frames")
data.to('deepmd/npy', 'init_data/graphene_h2_aimd')
```

Three lines. `box.npy`, `coord.npy`, `energy.npy`, `force.npy`, `type.raw`, `type_map.raw`. All ready for `init_data_sys`.

```{admonition} Pro Tip
:class: tip
DeePMD requires all frames in a single system directory to have the same atom count and type ordering. If you have mixed atom counts (72-atom bare slabs and 74-atom slab+H2), split them into separate directories. Mix them in one directory and DeePMD-kit will crash with a shape mismatch error that does not clearly tell you what went wrong.
```

---

## Connecting to dpgen

Whatever method you used, the output format is the same. Each system is a directory with npy files:

```
init_data/
├── set_72atoms/       # bare graphene slab frames
│   ├── set.000/
│   │   ├── box.npy, coord.npy, energy.npy, force.npy
│   ├── type.raw
│   └── type_map.raw
├── set_74atoms/       # graphene + 1 H₂ frames
│   ├── set.000/
│   │   ├── box.npy, coord.npy, energy.npy, force.npy
│   ├── type.raw
│   └── type_map.raw
```

Point to them in your `dpgen run` param.json:

```json
"init_data_sys": [
    "init_data/set_72atoms",
    "init_data/set_74atoms"
]
```

dpgen trains on ALL of `init_data_sys` from iteration 0. New data from active learning goes into `iter.*/02.fp/data.*` separately. <mark class="key-insight">The init data stays in the training set forever.</mark> It is the foundation. Every iteration builds on top of it.

So if your init data is bad (wrong forces, wrong energies, corrupted files), it poisons every single iteration. There is no mechanism to flush it out. The model retrains on init data every time. Bad foundation, bad building. Check your init data before you launch the loop.

```{admonition} Common Mistake
:class: caution
Each `init_data_sys` directory must have frames with the **same atom count and element ordering**. Mixing 72-atom and 74-atom frames in one directory will crash DeePMD-kit. The error message will mention array shape mismatches. It won't tell you "hey, your atom counts don't match." You have to figure that out yourself.
```

---

## Practical Reality: What We Actually Did

For the graphene + H2 project, we used the manual approach. Graphene is not "bulk." `init_surf` constructs surfaces by cleaving along Miller indices, but graphene is not cleaved from a bulk phase. It is a 2D material. And neither tool places adsorbate molecules.

So we did it by hand:

1. Set up graphene slab + H2 in QE input files (manually positioned)
2. Ran short AIMD at 300 K (50-100 steps per configuration)
3. Extracted snapshots with dpdata
4. Separated into directories by atom count
5. Plugged into `init_data_sys`

Total effort: a day of setup, a few hours of DFT compute. Because we controlled the configurations, we ensured coverage of specific geometries: different H2 heights above the surface, different orientations, different loadings. That kind of targeted coverage is something `init_bulk` and `init_surf` cannot give you, because they do not know what "H2 approaching a graphene sheet" means. They were not designed for that problem. Forcing them to solve it would have wasted more time than doing it manually.

```{admonition} Bottom Line
:class: note
`dpgen init_bulk` and `dpgen init_surf` are great when your system fits the mold. Standard bulk crystals and clean surfaces. Use them. They will save you real time.

But don't force a square peg into a round hole. For 2D materials with adsorbates, amorphous structures, or anything that doesn't look like a textbook example, go manual. A day of hands-on curation beats a week of fighting tools that weren't designed for your problem.
```

---

## Takeaway

Initial training data is the seed that starts the dpgen loop. Too little and the model cannot run MD. Too much and you wasted DFT compute on structures the active learning loop would have chosen better. 50-200 frames around equilibrium is the sweet spot. `dpgen init_bulk` and `dpgen init_surf` automate generation for standard systems. For everything else, run short AIMD, convert with dpdata, and let the active learning loop handle the rest.

The init data does not need to be great. It just needs to be good enough to survive the first iteration. One iteration down. The loop takes it from there.
