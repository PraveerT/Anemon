# Q-Rot-Cycle Verdict — honest fixed-fusion re-eval

Date: 2026-05-28. Supersedes the optimistic framing in `RECENT_QROT_BREAKTHROUGHS.md`.

## TL;DR
The depth "q-rotation-cycle consistency" gain is **generic dropout-consistency
(R-Drop), not the quaternion rotation**. The CNXXL "92.116%" qrot fusion is
**noise**. The doc's numbers came from a **test-tuned fusion grid**.

## Method
- Battery: `experiments/qexp_queue.py` — clean / qrot(uniform SO(3)) /
  ident(rot_mode=z, angle=0 → pure dropout consistency, no rotation), 5 seeds
  {29,31,37,43,47}, faithful 260-epoch recipe from the best run. Plus amp scouts.
- Honest eval: `experiments/qexp_fixed_eval.py` — FG83 base + w·branch (temp 1),
  **fixed a-priori weight, NOT grid-searched on the 482-sample eval set.**
- Also reports the trainer's `log_prob_fusion` (2989-config test-tuned grid) for
  contamination size.

## Paired result (all 5 seeds)
| comparison | branch-solo | fused w=0.75 | fused w=1.0 | seeds + |
|---|---:|---:|---:|---:|
| ident − clean (R-Drop, no rotation) | +2.07 | +0.33 | +1.12 | 4/5 |
| qrot − clean (uniform SO(3)) | +2.78 | +0.33 | +0.95 | 3/5 |
| **qrot − ident (the rotation itself)** | +0.71 | **+0.00** | **−0.17** | **2/5** |

- The real lift = consistency regularization (ident beats clean, 4/5 seeds).
- The rotation adds 0.00/−0.17 fused, 2/5 seeds → decorative.
- Amplification (rot-weight 0.02→0.16, small-so3@45°, z@60°) does not pull qrot
  clear of ident → nothing to amplify.

## Contamination
Test-tuned grid inflates clean fused from ~83.9 (honest w=1.0) / 84.8 (best
fixed) up to 85.7. The +1.7pp inflation + per-epoch test-noise reshuffling
manufactured the doc's "+0.456pp, 5/5 quaternion gain."

## CNXXL 92.116 (separate check)
444/482 occurs at only 4.2% of fusion-weight space; qrot z+2 solo = 430/482
(89.2%) < cnxxl 440 (91.3%). The "gain" is one sample inside a tuned sliver. Not
a defensible claim.

## Mechanism
model.train() keeps dropout on; corr_cycle = rotate-then-inverse-rotate ≈ the
original cloud, so 2 of the 3 KL terms are pure dropout consistency. On 1050
train samples, R-Drop alone regularizes. Constructive direction: tune the
consistency term (dropout, weight) directly — not quaternions.
