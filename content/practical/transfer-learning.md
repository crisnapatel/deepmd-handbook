# Transfer Learning

Training a model from scratch takes 1,000,000 steps. Transfer learning? 400,000. Same accuracy. That is not a rounding error. That is slashing your GPU bill in half.

And if you're on an HPC cluster where GPU hours are rationed like lab equipment during finals week, this is the difference between finishing your project this semester and writing an embarrassing supplemental allocation request. I've been on both sides of that. Trust me on this one.

---

## What Transfer Learning Actually Is

Picture this. Every dpgen iteration, four neural networks get initialized with **random weights** and train from scratch. Every. Single. Iteration. The model from iteration 3? Gone. Iteration 4 starts over with random noise where knowledge used to be.

Think about that for a second.

By iteration 5, the model already knows 95% of what it needs to know. Carbon is carbon. Hydrogen is hydrogen. The C-C bond in graphene is 1.42 A. The H-H bond is 0.74 A. The model learned all of that in iterations 0 through 4. And then dpgen throws it in the trash and says "learn it again from nothing."

That's like making a student who already passed Physics 101 re-derive F=ma from first principles before they can start Physics 201. Every semester. From scratch. The student already knows mechanics. Let them build on it.

Transfer learning says: **<mark class="key-insight">stop throwing away the previous model.</mark>** Instead of random initialization, start from the weights trained in the previous iteration. The network already "knows" the general shape of the potential energy surface. It just needs to absorb the 50-100 new configurations that the `fp` step just labeled.

The result: faster convergence, fewer training steps, lower compute cost. Same accuracy in less than half the wall time.

This is the part that made it click for me. Those first 5 iterations do the heavy lifting. Everything after that is refinement. And refinement does not require starting from zero.

---

## How dpgen Implements It: `training_reuse_iter`

One field in your `param.json` flips the switch. `training_reuse_iter` tells dpgen: "starting from this iteration number, reuse the model from the previous iteration instead of training from scratch."

Here's the config from a real graphene + H2 `param.json`. Pull up your param.json and compare.

```json
"training_reuse_iter": 5,
"training_reuse_numb_steps": 400000,
"training_reuse_start_lr": 0.0001,
"training_reuse_start_pref_e": 1.0,
"training_reuse_start_pref_f": 100,
"training_reuse_old_ratio": "auto"
```

Six fields. Let me walk through every single one.

### `training_reuse_iter`: 5

Iterations 0 through 4 train from scratch. Iteration 5 onward reuses the previous model.

Why not from iteration 0? Because **the initial model is random garbage.** There is nothing worth reusing. You need a few rounds of honest, from-scratch training so the model develops a real representation of your system. Trying to do transfer learning on a model that doesn't know anything yet is just training from scratch with extra overhead.

Five is a solid default. By that point, the model has seen enough DFT-labeled configurations to have a baseline worth preserving. Simple system? Maybe 3. Multi-element system with lots of configurations? Try 7 or 8. Start there. Adjust later.

### `training_reuse_numb_steps`: 400,000

During transfer learning iterations, train for 400k steps instead of the full million. The model already has good weights. It does not need a million steps to converge. Just enough to absorb the new data and fine-tune.

This is where the savings actually live. Let me put real numbers on it. 400k steps at ~0.03 seconds/step on one GPU is about 3.3 hours. Versus ~8.3 hours for a million steps. Multiply that by 4 models per iteration, times 10-20 iterations. You are saving real time. Not hypothetical time. Real wall-clock hours you can use for something else.

### `training_reuse_start_lr`: 0.0001

A lower starting learning rate. 10x smaller than the default 0.001. Not optional. Not a suggestion.

Think about it. The model already has good weights. If you slam it with a high learning rate, you destroy the learned representations in the first few thousand steps. The network "forgets" everything it knew and has to relearn it. You just defeated the entire purpose of transfer learning.

0.0001 makes gentle corrections. The model nudges itself toward fitting the new data without bulldozing the old knowledge.

````{admonition} Catastrophic Forgetting
:class: warning
This is a real phenomenon in neural network fine-tuning. Not just a scary name. If the learning rate is too high, the model overwrites weights that encoded previous knowledge with weights optimized only for the new batch. Great performance on the new data, terrible performance on everything else.

I learned this the expensive way. Set `start_lr` to the default 0.001 on a transfer learning run, watched the loss spike at step 0, and spent an hour wondering what broke. The model had basically reset itself. Wiped clean. All that previous training, gone in the first few hundred steps. The lower learning rate is the defense. Don't skip it.

```
# What I saw in lcurve.out when I got the LR wrong:
# step 0:    loss = 84.3   <-- the model FORGOT everything
# step 1000: loss = 12.1   <-- relearning from scratch
# ...that's not transfer learning, that's amnesia
```
````

### `training_reuse_start_pref_e` and `training_reuse_start_pref_f`

Loss function prefactors during transfer learning.

- `start_pref_e: 1.0`: weight on energy errors
- `start_pref_f: 100`: weight on force errors

Forces get 100x the weight of energies. This is standard DeePMD practice. Forces provide much richer per-atom gradient information than a single total energy number. During fine-tuning, you especially want the model to nail the forces on the new configurations, because that is what determines whether your MD simulation stays alive or launches atoms into the void.

These may differ from your initial training prefactors. That is intentional. Fine-tuning has different priorities than from-scratch training.

### `training_reuse_old_ratio`: "auto"

How dpgen balances old versus new training data during transfer learning.

Set it to `"auto"` and dpgen figures out a reasonable ratio based on dataset sizes. You don't want the model to only see new data (catastrophic forgetting again). You also don't want the old data to drown out the new data (no learning). `"auto"` finds the middle ground.

If you need manual control, set this to a float between 0 and 1 (e.g., 0.7 means 70% old data, 30% new). But `"auto"` works in practice. Don't overthink it.

---

## The Full Picture

Here is what a dpgen run looks like with transfer learning enabled:

| Iteration | Training Mode | Steps | Starting LR | Model Init |
|-----------|--------------|-------|-------------|------------|
| 0 | From scratch | 1,000,000 | 0.001 | Random weights |
| 1 | From scratch | 1,000,000 | 0.001 | Random weights |
| 2 | From scratch | 1,000,000 | 0.001 | Random weights |
| 3 | From scratch | 1,000,000 | 0.001 | Random weights |
| 4 | From scratch | 1,000,000 | 0.001 | Random weights |
| 5 | **Transfer** | **400,000** | **0.0001** | Previous model |
| 6 | **Transfer** | **400,000** | **0.0001** | Previous model |
| ... | **Transfer** | **400,000** | **0.0001** | Previous model |

Iterations 0-4: full training from nothing. Iterations 5+: the model builds on what it already knows. If your dpgen run goes 20 iterations, you just saved 15 iterations worth of training time. Real GPU hours. Real money (or real allocation units, same thing).

---

## When Transfer Learning Shines

Transfer learning works best when the new data is **similar but not identical** to what the model already knows:

- **Same chemistry, new thermodynamic conditions.** Trained at 300 K, now adding data from 500 K. The bonding is the same, just more thermal motion. The existing weights are a perfect starting point.
- **Same elements, slightly different environments.** H2 at 3 A above graphene vs. H2 at 2.5 A. The model already knows C-H and H-H interactions. It just needs to learn the closer approach.
- **Adding data from new dpgen iterations.** The most common case. Each iteration adds a handful of new configurations that are variations on what the model already knows.
- **Same system, different cell sizes.** Trained on a 2x2 supercell, now adding 3x3 data. Local chemistry is identical, just more atoms.

In all of these cases, the model's existing weights are a legitimate head start. The new data does not require fundamentally different representations, just refinements.

## When It Doesn't

Be honest with yourself here. Transfer learning is not magic. I've seen people waste more time debugging a bad warm start than they would have spent just training from scratch.

- **Completely different elements.** A model trained on C and H knows nothing useful about Si and O. The learned weights encode C-H chemistry. Reusing them for silica is starting from a bad place, not a good one. You are not saving time. You are adding confusion.
- **Wildly different energy scales.** If the previous model was trained on bulk structures at -278 eV/atom and you suddenly feed it isolated molecules at -16 eV/atom, the existing weights are calibrated to the wrong scale. Transfer learning will fight itself trying to reconcile the two. (See the [energy scale traps](energy-scale.md) page for why this gap is so destructive.)
- **The previous model was bad.** If iteration 4 produced a terrible model (high loss, unstable MD), reusing it for iteration 5 just propagates bad weights forward. Garbage in, garbage out. Sometimes you need a clean start.
- **You changed the model architecture.** Different `sel`, `neuron`, or `rcut` between iterations means the weight tensors have different shapes. You literally cannot load the old weights into the new architecture. dpgen falls back to fresh training automatically, but know that it is happening.

| Scenario | Transfer learning? | Why |
|----------|-------------------|-----|
| New dpgen iteration, same system | Yes | Most common case; small incremental changes |
| Higher temperature, same chemistry | Yes | Bonding is the same, just more thermal motion |
| Completely different elements (C+H to Si+O) | No | Learned weights encode wrong chemistry |
| Wildly different energy scales | No | Weights calibrated to wrong scale |
| Previous model was bad (high loss) | No | Propagates bad weights |
| Changed model architecture (`sel`, `rcut`) | Can't | Weight tensors have incompatible shapes |

```{admonition} Honest Assessment
:class: caution
Transfer learning is a warm start. If the warm start is pointing in the wrong direction, you will spend more steps correcting course than you would have spent starting from scratch. When in doubt, compare `lcurve.out` from a transfer learning iteration against a from-scratch iteration on the same data. If the transfer learning loss starts higher or converges slower, your warm start is not helping. It is hurting.
```

---

## Practical Monitoring: lcurve.out

Here is how you verify that transfer learning is actually doing its job. After each training run, open that `lcurve.out`. That is the file DeePMD-kit writes with training and validation loss at each step.

**From-scratch iteration (iteration 2)**, loss starts high and grinds down over 1M steps:

```
# step      l2_tst    l2_trn    l2_e_tst  l2_e_trn  l2_f_tst  l2_f_trn
      0    8.43e+01  7.82e+01  2.11e+00  1.96e+00  8.43e+01  7.82e+01
  10000    2.15e+00  2.08e+00  4.32e-02  3.89e-02  2.15e+00  2.08e+00
 100000    3.45e-01  3.21e-01  8.12e-03  7.54e-03  3.45e-01  3.21e-01
1000000    8.72e-02  7.95e-02  2.15e-03  1.98e-03  8.72e-02  7.95e-02
```

**Transfer learning iteration (iteration 7)**, loss starts LOW and refines further in just 400k steps:

```
# step      l2_tst    l2_trn    l2_e_tst  l2_e_trn  l2_f_tst  l2_f_trn
      0    1.24e-01  1.15e-01  3.82e-03  3.51e-03  1.24e-01  1.15e-01
  10000    1.08e-01  9.87e-02  3.14e-03  2.89e-03  1.08e-01  9.87e-02
 100000    7.65e-02  7.12e-02  2.43e-03  2.21e-03  7.65e-02  7.12e-02
 400000    6.89e-02  6.34e-02  1.98e-03  1.82e-03  6.89e-02  6.34e-02
```

See it? The transfer learning iteration starts at step 0 with a loss of ~0.12. Not ~84. That is the previous model's knowledge carrying over. It refines down to ~0.07 in just 400k steps. The from-scratch run needed all 1M steps to reach ~0.08.

That first number at step 0 is your sanity check. If it is low, transfer learning is working. If it is high, something went wrong.

**The red flag:** If your transfer learning iteration starts with a loss **higher** than where the previous iteration ended, stop. Check your learning rate. Check your data paths. Check that dpgen is actually loading the previous model and not a random initialization. Something is broken. Pay attention to this next part: a loss that spikes at step 0 of a transfer learning iteration almost always means the learning rate is too high or the model weights did not load correctly. Those are the only two causes I have ever encountered.

---

## Putting It in Your param.json

Here is a minimal addition to your existing `param.json`. You do not need to restructure anything. Just add the `training_reuse_*` keys alongside your existing training configuration:

```json
{
    "training": {
        "numb_steps": 1000000,
        "seed": 10,
        "disp_freq": 1000,
        "save_freq": 10000,
        "start_lr": 0.001,
        "stop_lr": 3.51e-08,
        "start_pref_e": 0.02,
        "start_pref_f": 1000,
        "limit_pref_e": 1,
        "limit_pref_f": 1
    },
    "training_reuse_iter": 5,
    "training_reuse_numb_steps": 400000,
    "training_reuse_start_lr": 0.0001,
    "training_reuse_start_pref_e": 1.0,
    "training_reuse_start_pref_f": 100,
    "training_reuse_old_ratio": "auto"
}
```

<mark class="silent-fail">The `training_reuse_*` keys go at the **top level** of `param.json`, not inside the `training` block.</mark> dpgen reads them separately and overrides the training configuration when the current iteration number meets or exceeds `training_reuse_iter`. Put them inside the `training` block and dpgen ignores them. No error. No warning. Just full million-step training every iteration, and you wondering why your run is taking twice as long as expected.

Read that again. Seriously. Top level. Not nested.

Six lines. That is the whole change.

---

## Takeaway

Transfer learning in dpgen is a straightforward optimization with outsized payoff. Set `training_reuse_iter` to 5, drop the step count to 400k, lower the learning rate to 0.0001, and let dpgen handle the rest. The model reuses what it already knows and spends its limited training budget on what is actually new.

But do not treat it as a silver bullet. Watch `lcurve.out`. Verify the loss starts low at step 0. And if you are switching to fundamentally different chemistry, skip transfer learning and train from scratch. A bad warm start is worse than a cold one. I've seen this go wrong too many times to sugarcoat it.
