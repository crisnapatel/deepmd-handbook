# QE vs VASP for First-Principles Labeling

Half the dpgen tutorials out there assume you are using VASP. Fair enough; VASP dominates the literature. But you are here because you are using Quantum ESPRESSO. Maybe it is free and open-source and your group does not have a VASP license. Maybe you do have VASP, but you want to understand what actually changes when you switch DFT engines under dpgen's hood. Either way, the dpgen docs do not make the differences obvious.

They should. Because the differences will silently break your workflow if you do not know where to look.

The core dpgen loop is identical regardless of DFT engine: explore with LAMMPS, identify uncertain configurations, label them with first-principles, retrain. But the configuration files (`param.json` and `machine.json`) differ in ways that range from "mildly annoying" to "three days of wasted compute." Let me trace through every difference that matters.

---

## The fp_style Switch

Everything starts with one field in `param.json`:

```json
"fp_style": "vasp"
```

or

```json
"fp_style": "pwscf"
```

Notice that. It is `"pwscf"`. Not `"qe"`. Not `"quantum-espresso"`. The name `pwscf` comes from QE's plane-wave DFT code `pw.x`, historically called PWscf. dpgen uses the old name. If you type `"qe"`, dpgen does not recognize it. No helpful error message. Just a traceback that does not point at the actual problem.

Here is what nobody tells you: this single field changes four things at once.

- How dpgen generates input files for each DFT calculation
- How it reads output (forces, energies, stresses)
- How it handles pseudopotentials
- What command it runs

One string. Four consequences.

---

## param.json: VASP Configuration

```{admonition} Config Walkthrough
:class: note
VASP users specify a separate INCAR file and POTCAR files. dpgen reads the INCAR, generates POSCAR from LAMMPS snapshots, concatenates POTCARs in `type_map` order, and creates KPOINTS. The `fp_pp_files` list must match `type_map` order exactly. Wrong order means wrong pseudopotential on each element. VASP won't complain. The SCF might even converge. The physics will be garbage.
```

```json
{
    "type_map": ["C", "H"],
    "fp_style": "vasp",
    "fp_incar": "INCAR",
    "fp_pp_path": "/path/to/potcars",
    "fp_pp_files": ["POTCAR_C", "POTCAR_H"],
    "fp_aniso_kspacing": [0.5, 0.5, 0.5]
}
```

dpgen concatenates `POTCAR_C` then `POTCAR_H` into a single `POTCAR` in `type_map` order. That order is load-bearing. More on that below.

---

## param.json: QE Configuration

```{admonition} Config Walkthrough
:class: note
QE users don't provide a separate input file. Instead, all QE input parameters live inside `user_fp_params` in `param.json`. dpgen builds the full `pw.x` input file from this dictionary. The nested keys (`control`, `system`, `electrons`) map directly to QE's `&CONTROL`, `&SYSTEM`, `&ELECTRONS` namelists. Everything in one place. No separate INCAR to keep track of.
```

```json
{
    "type_map": ["C", "H"],
    "fp_style": "pwscf",
    "fp_pp_path": "pseudo",
    "fp_pp_files": ["C.pbe-n-kjpaw_psl.1.0.0.UPF", "H.pbe-rrkjus_psl.1.0.0.UPF"],
    "user_fp_params": {
        "control": {
            "calculation": "scf",
            "tprnfor": true,
            "tstress": true,
            "pseudo_dir": "./",
            "outdir": "./",
            "disk_io": "none"
        },
        "system": {
            "ecutwfc": 50,
            "ecutrho": 400,
            "input_dft": "PBE",
            "vdw_corr": "dft-d3",
            "dftd3_version": 4,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.01,
            "nosym": true
        },
        "electrons": {
            "conv_thr": 1e-6,
            "mixing_beta": 0.4,
            "electron_maxstep": 200
        },
        "kspacing": 0.5
    }
}
```

````{admonition} Config Walkthrough
:class: note
For a complete 4-element QE-based dpgen param.json (O, Na, Cl, H in NaCl solution), see the [ICTP 2024 tutorial](https://github.com/cesaremalosso/tutorial_ictp2024/tree/main/dpgen). It uses ONCV pseudopotentials, ecutwfc=80, and a 10-iteration progressive exploration schedule from 300 K to 1100 K and 1 to 50000 bar. Comparing it with our 2-element C+H config highlights how multi-element systems scale the complexity.
````

Alright, enough theory. Let me walk through the parts that trip people up.

**`fp_pp_files`**: Individual `.UPF` files, one per element, in `type_map` order. dpgen copies them into each task directory and sets `pseudo_dir` accordingly. Unlike VASP, there is no concatenation. Each element keeps its own pseudopotential file. But the ordering rule is the same: first file matches first element in `type_map`.

**`user_fp_params`**: A nested dictionary that maps directly to QE's namelist structure. `control` becomes `&CONTROL`. `system` becomes `&SYSTEM`. `electrons` becomes `&ELECTRONS`. dpgen translates this dictionary into QE's input format automatically. You write JSON, QE gets its namelists.

**`kspacing`**: Lives at the top level of `user_fp_params`, not inside any namelist. dpgen uses this value to auto-generate the `K_POINTS` card based on cell dimensions. One number. dpgen does the math.

**`tprnfor` and `tstress`**: These must be `true`. <mark class="hard-req">This is not optional.</mark>

Without `tprnfor`, QE does not print forces. Without `tstress`, QE does not print stresses. dpgen needs both to create training data. If these are missing, <mark class="silent-fail">dpgen silently produces training data with zero forces.</mark> Zero. Your model trains on that. It learns that every atom in every configuration experiences zero force. The training loss looks fine because the model quickly learns to predict zero everywhere. Your MD simulation? Atoms don't move. Or they explode. Depends on the initial velocities.

QE does not complain when you forget `tprnfor`. It just silently gives you garbage. Helpful.

I almost cannot believe this is a silent failure. But it is. I forgot to set `tprnfor = .true.` on a project once. dpdata happily converted the output. The forces were all zeros. I trained a model on zero forces. It took me two days to figure out why the model was predicting a flat energy surface everywhere. Two days. Because of a missing boolean.

---

## The input_dft Gotcha

```{admonition} Warning: Energy Scale
:class: danger
**QE 7.3.1 wants `input_dft = 'vdw-df2-b86r'`, NOT `'rev-vdw-df2'`.**

The label `'rev-vdw-df2'` is obsolete. QE 7.3.1 will accept it without error, but it maps to a **different functional** than what you intended. This is a silent failure. Your DFT energies and forces will be computed with the wrong functional, your training data will be wrong, and your model will learn wrong physics.

No warning. No error message. QE just does the wrong thing quietly.

If you're using van der Waals functionals, **always check the QE source code or documentation for the correct label in your specific QE version.** The labels change between versions. What worked in QE 6.x may silently do something different in QE 7.x.
```

<mark class="silent-fail">This is the kind of bug that costs months.</mark> You run dpgen. Everything converges. The numbers look reasonable. Your model seems fine. Then your collaborator tries to reproduce your results with a different QE version, and the energies do not match. Because the functional was different the entire time. No crash at any point. Nothing suspicious in any log file. Just wrong physics, front to back.

For the graphene + H2 project, we use PBE with DFT-D3(BJ) dispersion correction instead of a vdW-DF functional, partly to avoid this exact trap:

```json
"input_dft": "PBE",
"vdw_corr": "dft-d3",
"dftd3_version": 4
```

Boring? Yes. Reproducible? Also yes. I will take boring and reproducible every single time.

---

## The `-in input` Issue

```{admonition} HPC Reality
:class: warning
dpgen (via dpdispatcher) runs QE as `pw.x -in input`, NOT as `pw.x < input` (stdin redirect). If your `machine.json` command invokes `pw.x` without the `-in` flag, and the input isn't piped via stdin, QE just sits there. Waiting. For input that will never arrive. Your job burns CPU hours until it hits the walltime limit and gets killed by the scheduler. Make sure your machine.json command includes `-in`.
```

In your `machine.json`, the QE command looks like:

```json
{
    "command": "mpirun -np 64 pw.x -in input",
    "machine_type": "pbs"
}
```

For VASP it would be:

```json
{
    "command": "mpirun -np 64 vasp_std",
    "machine_type": "pbs"
}
```

VASP reads `INCAR`, `POSCAR`, `POTCAR`, `KPOINTS` from the working directory automatically. Does not need to be told where to look. QE reads a single input file and needs to be pointed at it explicitly. Different conventions, different failure modes.

---

## Pseudopotential Handling

This is a fundamental architectural difference between the two codes.

### VASP

VASP uses PAW datasets called POTCARs. One file per element. dpgen concatenates them in `type_map` order into a single `POTCAR` file:

```
potcars/
    POTCAR_C      # Carbon PAW dataset
    POTCAR_H      # Hydrogen PAW dataset
```

dpgen effectively runs `cat POTCAR_C POTCAR_H > POTCAR`. Order matters. If `type_map` says `["C", "H"]` but you list the files as `["POTCAR_H", "POTCAR_C"]`, every carbon atom gets hydrogen's pseudopotential. VASP will not complain. The SCF might even converge. The results will be completely wrong. And nothing in the output will tell you. This is the wrong-name-on-the-exam problem: every grade assigned to the wrong person, nothing crashes, everything is quietly wrong.

### QE

QE uses individual `.UPF` files. No concatenation. dpgen copies each file into the task directory and references them in the `ATOMIC_SPECIES` card:

```
pseudo/
    C.pbe-n-kjpaw_psl.1.0.0.UPF
    H.pbe-rrkjus_psl.1.0.0.UPF
```

The `fp_pp_files` list must match `type_map` order:

```json
"type_map": ["C", "H"],
"fp_pp_files": ["C.pbe-n-kjpaw_psl.1.0.0.UPF", "H.pbe-rrkjus_psl.1.0.0.UPF"]
```

Same rule, same consequence: wrong order, wrong pseudopotentials, wrong physics, silent failure.

```{admonition} Common Mistake
:class: caution
QE pseudopotential filenames are long and ugly. It is tempting to rename them to `C.UPF` and `H.UPF`. Don't. The full name encodes the functional (pbe), the pseudization method (kjpaw, rrkjus), and the library version (psl.1.0.0). When you inevitably switch pseudopotentials or compare results, you will want to know exactly which files you used. The verbose filename is your documentation. Future you will thank present you.
```

---

## K-Points

### VASP

VASP reads k-points from a `KPOINTS` file or uses `KSPACING` in the `INCAR`. dpgen can generate the `KPOINTS` file or pass `KSPACING` through `user_incar_params`.

### QE

dpgen generates the `K_POINTS` card from the `kspacing` value in `user_fp_params`:

```json
"kspacing": 0.5
```

dpgen computes the k-mesh dimensions from `kspacing` and the cell vectors, then writes:

```
K_POINTS automatic
nx ny nz  0 0 0
```

No separate file. No extra configuration. One number. dpgen handles the rest. This is actually simpler than the VASP approach.

---

## Side-by-Side Comparison

| Feature | VASP | QE (PWscf) |
|---------|------|------------|
| `fp_style` value | `"vasp"` | `"pwscf"` |
| Input specification | Separate `INCAR` file via `fp_incar` | Inline dict via `user_fp_params` |
| Pseudopotentials | POTCAR files, concatenated | Individual `.UPF` files, copied |
| PP specification | `fp_pp_files: ["POTCAR_C", "POTCAR_H"]` | `fp_pp_files: ["C.pbe-n-kjpaw_psl.1.0.0.UPF", "H.pbe-rrkjus_psl.1.0.0.UPF"]` |
| K-points | `KPOINTS` file or `KSPACING` in INCAR | `kspacing` in `user_fp_params` |
| DFT command | `mpirun vasp_std` | `mpirun pw.x -in input` |
| Input file read | Auto-reads from working dir | Explicit `-in input` flag |
| Force output | Always printed | Requires `tprnfor = true` |
| Stress output | Always printed (with `ISIF >= 1`) | Requires `tstress = true` |
| License | Commercial ($$$) | Free (GPL) |
| Typical speed | Faster for same system | Slightly slower |
| Scaling | License limits node count | No license limit |

---

## Performance and Licensing

Let me be blunt.

VASP is faster. For the same system, same functional, same k-mesh, VASP typically finishes 10-30% faster than QE. Decades of optimization for plane-wave PAW calculations. It shows.

But VASP costs money. A group license runs thousands of dollars per year. And here is the practical constraint that nobody talks about in the tutorials: your VASP license limits how many simultaneous jobs you can run. dpgen can submit dozens or hundreds of DFT labeling jobs per iteration. If your license only covers a few nodes, you bottleneck hard.

QE is free. Run it on every node in the cluster simultaneously. No license server. No usage tracking. No "license seats exceeded" message killing your jobs at 2 AM.

For dpgen workflows where you are submitting 50-200 labeling jobs per iteration, this matters more than raw per-job speed. A lot more.

```{admonition} HPC Reality
:class: warning
The real bottleneck in dpgen is almost never the DFT code's speed. It is how many DFT jobs you can run in parallel. A code that is 20% slower but can run on unlimited nodes finishes the iteration faster than a code that is 20% faster but limited to 4 concurrent jobs by licensing.

If you have a VASP license with generous node allocation, use VASP. If you are license-limited or license-free, use QE. The model does not care which DFT engine produced the training data. It cares that the data is correct and consistent.
```

---

## Converting Between VASP and QE Configs

Switching `fp_style` is not a find-and-replace job. Here is the full checklist. Skip a step and something will break silently. I've seen this go wrong too many times.

1. Change `fp_style` from `"vasp"` to `"pwscf"` (or vice versa)
2. Replace `fp_incar` with `user_fp_params` (or vice versa). Translate every INCAR tag to its QE equivalent. This is tedious. There is no shortcut
3. Replace POTCAR files with `.UPF` files (or vice versa). Make sure the functional matches
4. Update `fp_pp_path` and `fp_pp_files`
5. Update the DFT command in `machine.json`
6. Verify `tprnfor` and `tstress` are set (QE only; VASP does this automatically)
7. **Run a single-point test calculation on both codes with the same structure and compare energies/forces.** Not optional. Not a suggestion.

That last point is the one people skip because they are in a hurry. Different pseudopotentials, different basis set implementations, different default parameters. Even with "the same" functional, VASP and QE give slightly different absolute energies. The forces should agree to within ~1 meV/A for well-converged calculations. If they do not, your settings are not equivalent. Go back and figure out why before you commit to a 20-iteration dpgen campaign.

```{admonition} Common Mistake
:class: caution
Don't mix VASP-labeled and QE-labeled training data in the same model without verifying energy consistency. Different codes have different energy zeros, different pseudopotential constructions, and subtly different implementations of the same functional. Mixing them introduces systematic noise that the model can't learn around.

Pick one DFT code for your entire dpgen campaign. Stick with it. This is not a preference. It is a consistency requirement.
```

---

## Takeaway

The DFT engine is a labeling tool. dpgen does not care whether the labels come from VASP or QE. It cares that they are accurate and consistent.

If you are using QE, three things will actually bite you: `fp_style` must be `"pwscf"` (not `"qe"`), `tprnfor` and `tstress` must be `true`, and you need to double-check your `input_dft` label against your specific QE version. Everything else is bookkeeping. Important bookkeeping, but bookkeeping.

Get those three things right and the rest falls into place.

## QE Fundamentals

If you are coming from VASP and need a QE refresher, the [CSI Princeton Workshop (Session 2)](https://github.com/cesaremalosso/workshop-july-2023/tree/main/hands-on-sessions/day-1/2-quantum-espresso) covers QE input anatomy, ecutwfc/kpoint convergence testing, and vc-relax workflows with ASE integration.
