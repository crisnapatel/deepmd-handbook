# Gap-Filling: Manual Data Curation

Let me tell you what happened. dpgen converged. 98% accurate. The model looked bulletproof. I was ready to write the methods section. Then I ran production MD with 20 H2 molecules on graphene instead of the usual 4, and the model immediately launched two hydrogen atoms through the graphene sheet like it was a hologram.

No crash. No error. No model deviation spike. <mark class="silent-fail">The model confidently predicted that atoms can phase through carbon sheets.</mark> Because why not? Nobody told it otherwise. It had never seen two hydrogen atoms pushed close together by crowding. It had never encountered the repulsive wall at short distances. So when the simulation wandered into those configurations, the model did what neural networks do when they extrapolate: it made something up. Smoothly. Confidently.

That is the gap-filling problem. And here is why dpgen alone cannot solve it. dpgen explores configuration space by running MD, which means it only finds structures that dynamics can reach from your starting points. High barriers? Rare events? Specific geometries that 300 K NVT will never stumble into? dpgen has no idea those exist. The Boltzmann distribution is working against you. Equilibrium MD exponentially suppresses high-energy configurations. The model never sees them, never learns them, and then confidently hallucinates when it encounters them in production.

You have to go find those gaps yourself.

```{admonition} Why Our Tutorial Examples Don't Need Gap-Filling
:class: note
Our Ar model was trained on two phases (FCC + liquid), covering a range of configurations. Our water model uses ICTP pre-computed data spanning liquid water conditions. Neither system has the kind of missing physics (close contacts, reactions, desorption events) that requires manual gap-filling. But the moment you study adsorption on surfaces, reactions, phase transitions between very different structures, or any system where MD can't sample the full relevant configuration space, gap-filling becomes essential.
```

## When You Need Gap-Filling

You will know. The symptoms are unmistakable.

1. **Production MD crashes at conditions dpgen never explored.** The model was rock solid at 300 K but detonates at 77 K. Or it handles 4 H2 molecules but falls apart with 20. dpgen never sent the model on a field trip to those conditions. No training data. No learning. The model is flying blind and does not know it.

2. **Model deviation spikes in regions you actually care about.** You run a test simulation and see sudden jumps in deviation, even though dpgen declared convergence three iterations ago. The model is telling you, loudly, that it is guessing. Listen to it.

3. **Unphysical behavior during dynamics.** H2 molecules passing through graphene. Bond lengths hitting 0.5 Angstroms. Atoms sitting on top of each other. If the model has never seen repulsive close-contact configurations, it does not know those should cost a fortune in energy. It just lets atoms walk through each other like they are not even there.

4. **Missing physics that MD cannot sample.** Your dpgen exploration only ran NVT. It never saw strained structures. Never sampled different cell sizes. Never explored specific adsorption sites that require placing an H2 molecule exactly where you want it. You cannot wait for dynamics to find these. Dynamics never will.

```{admonition} Key Insight
:class: tip
dpgen explores configuration space **dynamically**. It runs MD and picks uncertain frames. But MD is biased toward low-energy regions of the potential energy surface. High-energy configurations (close contacts, strained cells, transition states) are exponentially unlikely to show up in equilibrium MD, even at high temperatures. The Boltzmann distribution is working against you.

Gap-filling supplements dynamic exploration with **targeted static sampling**. You deliberately build the configurations the model needs to see. You go to the gaps instead of waiting for dynamics to find them. Dynamics never will.
```

---

## The Gap-Filling Workflow

Four steps. Each one has a specific job. Let me trace through them with real examples from the graphene + H2 project.

### Step 1: Generate Candidate Structures with Classical Force Fields

Before you burn DFT compute, use cheap classical potentials to generate a diverse pool of structures. The classical FF does not need to be accurate. I am serious. It does not even need to be good. It just needs to produce physically reasonable structures that span the configuration space you want to fill.

Think of it like scouting. You are not asking the classical FF to solve your problem. You are asking it to show you what different parts of configuration space look like. Where are the atoms when you have 20 H2 on graphene instead of 4? What does a crowded surface look like? What does H2 gas look like at high density? The classical FF answers those structural questions cheaply. DFT answers the energetics questions expensively. You use the cheap answer to decide where to spend the expensive answer.

For our project:

| Material | Force Field | Runs | Systems | Time per run |
|----------|------------|------|---------|--------------|
| Graphene | AIREBO | 19 | bare, +4/8/12/20 H2, H2 gas | ~1 min |
| Graphanol | ReaxFF | 18 | bare, +2/4/6/8/12 H2, H2 gas | ~30 min |
| Graphamine | ReaxFF | 18 | bare, +2/4/6/8/12 H2, H2 gas | ~30 min |

```console
$ bash graphene_h2/gap_filling/lammps/run_all.sh
```

These simulations use `replicate` to create larger supercells: 2x2x1 for graphene (288 atoms from 72), 2x3x1 for graphanol/graphamine (432 atoms from 72). Bigger cells sample more configurations per trajectory and give the FPS selector (coming in step 2) more material to work with.

Temperatures span 77 K to 800 K. Not optional. Not a suggestion. Low temperatures sample near-equilibrium structures. The atoms barely move. The model probably already knows this regime. High temperatures shove atoms into unusual positions. Compressed bonds. Atoms crowded together. Configurations your model has never imagined. You need both ends of the spectrum, and you need everything between.

```{admonition} HPC Reality
:class: warning
AIREBO runs in seconds per trajectory. ReaxFF takes tens of minutes. Neither needs a GPU. Run these on your local workstation or a single HPC node. Do not waste GPU allocation on classical FF runs. Save those hours for DFT.
```

### Step 2: Farthest Point Sampling (FPS)

You now have thousands of frames from those classical MD runs. You obviously cannot run DFT on all of them. That would defeat the entire purpose. You need to select a maximally diverse subset. The smallest set of frames that covers the largest swath of configuration space.

Enter **Farthest Point Sampling**. This is the part that made it click for me.

The idea is almost embarrassingly simple. Imagine you are placing fire stations in a city. You want each station as far from every other station as possible, so the coverage is maximal and the overlap is minimal. FPS does this in configuration space:

1. Start with one random frame from your pool.
2. For every remaining frame, compute its "distance" to the nearest already-selected frame.
3. Select the frame that is farthest from anything already selected.
4. Repeat until you have enough frames.

That is the whole trick. The "distance" is computed in descriptor space (SOAP or a similar structural fingerprint). Two frames that look similar in descriptor space are neighbors. Two frames with very different atomic environments are far apart.

The result: a subset that maximally spans your configuration space. No clustering in one region. No huge blind spots elsewhere. Every selected frame is as different as possible from every other selected frame.

```console
$ conda run -n Analysis python3 common/scripts/fps_select_lammps.py \
    --material graphene --n_select 200

$ conda run -n Analysis python3 common/scripts/fps_select_lammps.py \
    --material graphanol --n_select 150
```

200 frames out of thousands. That is your DFT budget for this round.

```{admonition} Key Insight
:class: tip
Why FPS instead of random selection? Because random sampling is lazy. If 90% of your MD frames are near-equilibrium graphene at 300 K, random sampling hands you 90% near-equilibrium graphene. You already have plenty of that. You need the weird stuff. The edges. The configurations the model has not seen.

FPS deliberately avoids redundancy. It reaches out to the fringes of your configuration space and grabs the most alien frames it can find. For gap-filling, that is exactly the point.
```

### Step 3: QE Static Calculations

Now you have your selected configurations. FPS-selected frames from LAMMPS MD, plus manually designed static scans that cover specific physics MD might miss. Submit everything to QE for DFT labeling.

The FPS-selected frames cover broad configuration space. The static scans are surgical. Each one targets a specific gap in the model's knowledge:

| Scan Type | Configs | Atoms | Why |
|-----------|---------|-------|-----|
| H2 bond length | 14 | 2 | The model needs to know the intramolecular H-H potential |
| H2 height (horizontal) | 9 | 72-74 | Adsorption energy vs. distance from surface |
| H2 height (vertical) | 9 | 72-74 | Different orientation, different interaction |
| Strained bare slab | 4 | 72 | Mechanical response of the substrate |
| Strained slab + 1 H2 | 2 | 73-74 | Coupling between strain and adsorption |
| Bulk H2 gas | 6 | 16-64 | Dense gas-phase H2 interactions |
| Bond dissociation | 8 | 72 | O-H / N-H breaking (graphanol/graphamine only) |

These are not random. Each row exists because the model failed at something specific and this scan fixes it. The H2 bond length scan exists because the model got the H-H equilibrium distance wrong. The height scans exist because the adsorption curve had the wrong shape. The strained slab exists because the model saw only relaxed cells and did not know what a compressed lattice feels like. Every scan is a targeted response to a known failure.

```console
$ qsub graphene_h2/gap_filling/qe_scans/submit_qe.pbs
$ qsub graphene_h2/gap_filling/fps_selected/submit_qe.pbs
```

```{admonition} HPC Reality
:class: warning
The static scans (small systems, 2-74 atoms) finish in minutes each. The FPS-selected configs (128-448 atoms) can take 30-60 minutes each on 64-128 cores. For 200 FPS-selected configs at roughly 1 hour each, that is 200 CPU-hours on top of your dpgen budget. Not free. But compared to the weeks of production MD you would waste on a model with gaps, it is cheap insurance.
```

### Step 4: Curate and Integrate

QE jobs done. Now convert the outputs to DeePMD format and fold them into your training data:

```console
$ apptainer exec deepmd-dpgen.sif python3 \
    common/scripts/curate_gap_filling.py --all
```

This groups frames by atom count (<mark class="hard-req">DeePMD requires consistent atom counts within a dataset</mark>), creates the `set.000/` directories with `box.npy`, `coord.npy`, `energy.npy`, `force.npy`, and drops them in `init_data/`.

Then you add these new datasets to your `param.json`:

```json
"init_data_sys": [
    "set_2atoms",
    "set_4atoms",
    "set_8atoms",
    "set_72atoms",
    "set_74atoms",
    "set_gap_128atoms",
    "set_gap_288atoms",
    "set_gap_2atoms",
    "set_gap_320atoms",
    "set_gap_352atoms",
    "set_gap_384atoms",
    "set_gap_448atoms"
]
```

See those `set_gap_*` entries? That is your gap-filled data. It lives alongside the original AIMD-derived data in `init_data/`. When dpgen starts, it trains on ALL of `init_data_sys` from iteration zero. The gap-filled data is baked in from the very beginning. The model sees those configurations on day one. Not iteration five. Not after it has already developed bad habits. Day one.

Clean.

---

## Verifying the Gap Fill

You added data. But did you actually fill the gap you were aiming at? Here is where it gets interesting. Adding data is easy. Adding the *right* data is the whole challenge. Trust but verify.

```console
$ apptainer exec deepmd-dpgen.sif python3 \
    common/scripts/data_to_xyz.py --all

$ conda run -n Analysis python3 \
    common/scripts/visualize_training_data.py --all
```

Three things to look for:

- The energy distribution should now cover the range you were missing. If you added close-contact configurations, you should see high-energy points in the distribution that were not there before. If those points are missing, the gap is still open.
- Configuration space coverage (H2-surface distance distribution, for example) should span your target range. No visible holes. No empty bins where you expected data.
- The new data should not have introduced an energy scale mismatch. This one will bite you.

```{admonition} Common Mistake
:class: caution
You add gap-filling data and re-run dpgen, feeling good about yourself. But the new data has a wildly different energy scale than the existing data (see [Energy Scale Traps](energy-scale.md)). The model tries to fit both scales and fits neither.

For our project, the isolated H2 bond-length scan (2 atoms, roughly -16 eV/atom) sat on a completely different energy scale than the graphene slab data (roughly -278 eV/atom). That is grading kindergarten and PhD students on the same curve. Mixing them naively degraded the slab accuracy without meaningfully improving H2. You have to handle this deliberately. Check per-atom energies before you merge datasets. Always.
```

---

## When to Gap-Fill vs. When to Adjust dpgen

Not every problem needs gap-filling. Sometimes dpgen can handle it if you give it the right settings. Here is how to tell the difference:

| Situation | Fix | Why |
|-----------|-----|-----|
| Model deviation high in regions dpgen already explored | Adjust `trust_lo`/`trust_hi` or add more dpgen iterations | dpgen is designed for this; let it do its job |
| Too few frames got labeled in explored regions | Increase `fp_task_max` and re-run | dpgen found the candidates but hit the labeling budget |
| Model fails in conditions dpgen never explored | Gap-fill with targeted data | New temperatures, new compositions, different cell sizes; dpgen cannot find what it never looked for |
| Model fails at specific geometries | Gap-fill with static scans | Close contacts, transition states, bond dissociation; MD at any temperature is unlikely to sample these |
| Training data is fundamentally missing a region | Gap-fill | No strained structures, no high-pressure data, no surface with more than 4 adsorbates |

The rule of thumb: <mark class="key-insight">dpgen fills gaps in the regions it explores. Gap-filling fills gaps in the regions dpgen never visits.</mark> If MD can reach it, dpgen will eventually find it. If MD cannot reach it, you have to build it by hand.

## Takeaway

dpgen is an exploration engine. A very good one. But it explores by running dynamics, which means it only finds what dynamics can reach from your starting structures. High barriers, rare events, extreme conditions, specific geometries that equilibrium MD will never stumble into? Those gaps are yours to fill.

FPS gives you the most diverse subset from cheap classical MD. Static scans target specific physics the model gets wrong. Together with dpgen, you get comprehensive coverage. Without gap-filling, you get a model that is confident in the regions it knows and dangerously creative in the regions it does not.

I've seen this go wrong too many times. Do the gap-filling.
