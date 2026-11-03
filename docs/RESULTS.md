# Results

All numbers below are produced by `scripts/make_results.py` from `results/eval_*.jsonl`. They are **simulation** results on held-out cucumbers (never used for training), with the rule-based skin model described in SIMULATION.md. They are not real-robot results.

Success = at least 90 % of the peelable skin removed, at most 3 % gouged, no handle collision. 95 % intervals are Wilson score intervals. 'Estimated geometry' means the scripted expert only knows a straight cylinder on the gripper axis with a radius guessed to about 1.5 mm, as a camera estimate would give.

## Training data

- v1 shapes: 84 scripted-expert episodes through the GELLO joint path, 78 successful and used (75146 frames at 24 Hz). Per class (successful / recorded): mini 11/12, standard 12/12, long 12/12, thick 10/12, tapered 11/12, curved 12/12, mixed 10/12.
  RDP stroke dataset `peel_stroke_zarr`: 710 single-stroke clips, 57769 frames.
- v2 real cucumber types: 0 scripted-expert episodes through the GELLO joint path, 0 successful and used (0 frames at 24 Hz). Per class (successful / recorded): pickling 0/0, persian 0/0, slicing 0/0, english 0/0, crooked 0/0, necked 0/0, realistic 0/0.

## Test set v1 (first shapes, seeds 9000+)

Policies here were trained on v1 demos only.

| Method | Episodes | Success | 95 % CI | Mean coverage | Mean gouge | Median peak force |
|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 35 | 29/35 (83 %) | 67 to 92 % | 93.1 % | 0.00 % | 23.1 N |
| Scripted expert, no force feedback | 35 | 31/35 (89 %) | 74 to 95 % | 93.5 % | 0.05 % | 25.0 N |
| Scripted expert, force feedback, estimated geometry | 35 | 29/35 (83 %) | 67 to 92 % | 92.7 % | 0.02 % | 23.9 N |
| Scripted expert, no force feedback, estimated geometry | 35 | 17/35 (49 %) | 33 to 64 % | 92.3 % | 12.14 % | 32.3 N |
| RDP (vision + pose + wrench) | 2 | 0/2 (0 %) | 0 to 66 % | 87.7 % | 0.00 % | 22.1 N |

By cucumber class (successes / episodes):

| Method | mini | standard | long | thick | tapered | curved | mixed |
|---|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 5/5 | 4/5 | 5/5 | 5/5 | 4/5 | 4/5 | 2/5 |
| Scripted expert, no force feedback | 5/5 | 4/5 | 5/5 | 4/5 | 4/5 | 5/5 | 4/5 |
| Scripted expert, force feedback, estimated geometry | 5/5 | 4/5 | 5/5 | 5/5 | 4/5 | 3/5 | 3/5 |
| Scripted expert, no force feedback, estimated geometry | 2/5 | 2/5 | 1/5 | 4/5 | 3/5 | 1/5 | 4/5 |
| RDP (vision + pose + wrench) | 0/2 | - | - | - | - | - | - |

By cucumber class (mean coverage):

| Method | mini | standard | long | thick | tapered | curved | mixed |
|---|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 92 % | 92 % | 97 % | 93 % | 94 % | 93 % | 91 % |
| Scripted expert, no force feedback | 93 % | 93 % | 97 % | 92 % | 94 % | 93 % | 93 % |
| Scripted expert, force feedback, estimated geometry | 92 % | 92 % | 96 % | 92 % | 94 % | 90 % | 91 % |
| Scripted expert, no force feedback, estimated geometry | 93 % | 93 % | 93 % | 92 % | 93 % | 90 % | 92 % |
| RDP (vision + pose + wrench) | 88 % | - | - | - | - | - | - |

Paired comparisons on the same cucumbers (exact McNemar test on success):

- expert_force vs expert_noforce, 35 cucumbers: expert_force wins 1, expert_noforce wins 3, p = 0.625.
- expert_force_nominal vs expert_noforce_nominal, 35 cucumbers: expert_force_nominal wins 14, expert_noforce_nominal wins 2, p = 0.00418.
- rdp vs expert_force_nominal, 2 cucumbers: rdp wins 0, expert_force_nominal wins 2, p = 0.5.

