# Context: the quaternion program

Stable briefing. Changes rarely. If a message contradicts this file, the message
wins and someone should update this file.

## The setting

Point-cloud gesture recognition on NVGesture. 1050 training clips, 482 test
clips, 25 classes. Input is a depth-derived point cloud, 32 frames, 512 points
sampled down to 172. Channels are image-plane `(u, v, depth, frame_index)`, NOT
metric xyz. Training runs 150 epochs on a single A6000, roughly 3 hours.

Two model families are live:

| model | params | best | last-10 |
|-------|--------|------|---------|
| `ladder_B6T` (baseline, ordinary convs) | 2.9922M | 90.46 | 89.32 +- 0.83 |
| `ladder_Q1` (`QuatLadder`, quaternion encoder) | 1.4204M | 90.66 | not yet measured |

`ladder_Q1` is 47.5 percent of B6T's parameters at equal or better accuracy.
That efficiency claim currently has NO parameter-matched non-quaternion control,
which is the largest hole in the program.

## The question the program is trying to answer

Does quaternion structure buy anything on this task, or is it decoration.

## What is settled: quaternions do not improve accuracy

Nine operand pairs were tried for a relative-rotation term `R = a (x) conj(b)`,
each against its own shuffled-pairing control. All null. Pairs included two
temporal scales, motion against orientation, learned operands, and curl against
geometry, strain, velocity, global curl, impulse, and angular impulse.

Mechanisms, not shrugs:

- **The rotation route is decorative.** Deleting the term changed 1 prediction
  out of 482 and improved accuracy by 0.21.
- **The feature route is content-independent.** Scale-matched random noise
  reproduced trained accuracy exactly, 83.40 both.
- **Hamilton is not a special bilinear form.** As a learnable bilinear map,
  random initialisation matched Hamilton initialisation, 55.33 against 56.87,
  and 16 unconstrained outputs beat both at 58.82.
- **Holonomy is null.** Path-ordered products against the commutative control,
  nine configurations: +0.36 -1.10 +0.77 +0.36 -1.37 -0.48 -1.99 -0.81 +0.67.
  Mean -0.40, sign unstable.
- **Rotation plus reconstruction is catastrophic.** Random rotation alone costs
  45.73, because NVGesture classes are defined by absolute direction.

## What is real: the invariance crossover

The one solid quaternion result. Exact invariance to a shared right frame is
worth **-9.75 when that frame carries the label** and **+31.72 when it carries
nuisance**. Same machinery, same data, sign flips with the role of the frame.
Under injected nuisance, ordinary components collapse from 62.78 to 19.71 while
the quaternion comparison loses 1.60. No finite network learns a continuous
group invariance from 1050 clips.

Plus a capability proof on synthetic data built to contain the relation: 49.25
against a control pinned at chance, 16.67.

## The open question: angle shrinkage

`models/quat_act.py`. Groups of 4 channels are read as a quaternion, the
rotation angle is shrunk by a learned threshold, the group norm is preserved. A
sparsity prior in rotation space rather than coordinate space.

- On `QuatLinear` features: **+2.84 +- 1.63 over eight paired seeds**, 7 of 8
  positive, about 4.9 standard errors.
- On ordinary channels, group-size sweep over 2 seeds: gelu 83.10, g2 80.29,
  g4 82.68, g8 82.47, g16 82.99. Flat from 4 upward, no peak at 4, nothing beats
  gelu, and seed spread grows 0.21, 0.42, 2.69, 2.70, 4.15.

Hypothesis under test: the activation pays on quaternion-structured features and
is noise on arbitrary channel groups.

Two controls were never run and should be: `g4_perm` (permute slot positions
inside each group) and `g4_stride` (group channels 4 apart). They test
quaternion specificity directly rather than through the size curve. About 20
minutes each.

## The augmentation result

Architecture-independent, and the largest single effect measured on this
dataset. `GpuAugmentor` flattens to `(B, T*N, C)` and runs temporal operations
along that flattened axis, so each slot becomes a blend of two adjacent frames.
That looks like a bug. It is not: correcting it to operate along `T` at matched
degradation trains 2.01 worse.

But the magnitude is wrong. Scaling only the three temporal knobs, 2 seeds:
off 82.16, **x0.25 84.44**, x1.0 current 83.10.

## Two rigs

- **fast rig**: small network, 172 points, 60 epochs, floor around 82, roughly
  20 minutes. Between-seed sd 0.93. Used for screening.
- **full rig**: the real models above, 150 epochs, about 3 hours. Between-seed
  sd about 1.2.

Results do not automatically transfer between them. The fast rig is far from
saturated, so effects measured there tend to shrink at 90.

## Constraints

One GPU job at a time. Sequential, never parallel. A full-rig arm is 3 hours, so
a proposal costing four arms costs half a day and needs to justify that.
