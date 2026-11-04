# Results

All numbers below are produced by `scripts/make_results.py` from `results/eval_*.jsonl`. They are **simulation** results on held-out cucumbers (never used for training), with the rule-based skin model described in SIMULATION.md. They are not real-robot results.

Success = at least 90 % of the peelable skin removed, at most 3 % gouged, no handle collision. 95 % intervals are Wilson score intervals. 'Estimated geometry' means the scripted expert only knows a straight cylinder on the gripper axis with a radius guessed to about 1.5 mm, as a camera estimate would give.

## Training data

- v1 shapes: 84 scripted-expert episodes through the GELLO joint path, 78 successful and used (75146 frames at 24 Hz). Per class (successful / recorded): mini 11/12, standard 12/12, long 12/12, thick 10/12, tapered 11/12, curved 12/12, mixed 10/12.
  RDP stroke dataset `peel_stroke_zarr`: 710 single-stroke clips, 57769 frames.
- v2 real cucumber types: 63 scripted-expert episodes through the GELLO joint path, 57 successful and used (54597 frames at 24 Hz). Per class (successful / recorded): pickling 9/9, persian 9/9, slicing 8/9, english 9/9, crooked 6/9, necked 9/9, realistic 7/9.
  RDP stroke dataset `peel_stroke_v2_zarr`: 1214 single-stroke clips, 100257 frames.

## Test set v1 (first shapes, seeds 9000+)

Policies here were trained on v1 demos only.

| Method | Episodes | Success | 95 % CI | Mean coverage | Mean gouge | Median peak force |
|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 35 | 29/35 (83 %) | 67 to 92 % | 93.1 % | 0.00 % | 23.1 N |
| Scripted expert, no force feedback | 35 | 31/35 (89 %) | 74 to 95 % | 93.5 % | 0.05 % | 25.0 N |
| Scripted expert, force feedback, estimated geometry | 35 | 29/35 (83 %) | 67 to 92 % | 92.7 % | 0.02 % | 23.9 N |
| Scripted expert, no force feedback, estimated geometry | 35 | 17/35 (49 %) | 33 to 64 % | 92.3 % | 12.14 % | 32.3 N |
| RDP (vision + pose + wrench) | 23 | 13/23 (57 %) | 37 to 74 % | 89.7 % | 0.00 % | 21.6 N |

By cucumber class (successes / episodes):

| Method | mini | standard | long | thick | tapered | curved | mixed |
|---|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 5/5 | 4/5 | 5/5 | 5/5 | 4/5 | 4/5 | 2/5 |
| Scripted expert, no force feedback | 5/5 | 4/5 | 5/5 | 4/5 | 4/5 | 5/5 | 4/5 |
| Scripted expert, force feedback, estimated geometry | 5/5 | 4/5 | 5/5 | 5/5 | 4/5 | 3/5 | 3/5 |
| Scripted expert, no force feedback, estimated geometry | 2/5 | 2/5 | 1/5 | 4/5 | 3/5 | 1/5 | 4/5 |
| RDP (vision + pose + wrench) | 0/5 | 3/3 | 3/3 | 2/3 | 2/3 | 2/3 | 1/3 |

By cucumber class (mean coverage):

| Method | mini | standard | long | thick | tapered | curved | mixed |
|---|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 92 % | 92 % | 97 % | 93 % | 94 % | 93 % | 91 % |
| Scripted expert, no force feedback | 93 % | 93 % | 97 % | 92 % | 94 % | 93 % | 93 % |
| Scripted expert, force feedback, estimated geometry | 92 % | 92 % | 96 % | 92 % | 94 % | 90 % | 91 % |
| Scripted expert, no force feedback, estimated geometry | 93 % | 93 % | 93 % | 92 % | 93 % | 90 % | 92 % |
| RDP (vision + pose + wrench) | 83 % | 92 % | 94 % | 91 % | 92 % | 91 % | 89 % |

Paired comparisons on the same cucumbers (exact McNemar test on success):

- expert_force vs expert_noforce, 35 cucumbers: expert_force wins 1, expert_noforce wins 3, p = 0.625.
- expert_force_nominal vs expert_noforce_nominal, 35 cucumbers: expert_force_nominal wins 14, expert_noforce_nominal wins 2, p = 0.00418.
- rdp vs expert_force_nominal, 23 cucumbers: rdp wins 1, expert_force_nominal wins 7, p = 0.0703.

## Test set v2 (real cucumber types, seeds 10000+)

Policies here were trained on v1 + v2 demos unless the method name says otherwise.

| Method | Episodes | Success | 95 % CI | Mean coverage | Mean gouge | Median peak force |
|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 35 | 31/35 (89 %) | 74 to 95 % | 96.4 % | 0.22 % | 24.0 N |
| Scripted expert, no force feedback | 35 | 28/35 (80 %) | 64 to 90 % | 97.4 % | 2.02 % | 29.4 N |
| Scripted expert, force feedback, estimated geometry | 35 | 11/35 (31 %) | 19 to 48 % | 77.9 % | 5.90 % | 36.3 N |
| Scripted expert, no force feedback, estimated geometry | 35 | 4/35 (11 %) | 5 to 26 % | 77.5 % | 20.27 % | 51.6 N |

By cucumber class (successes / episodes):

| Method | pickling | persian | slicing | english | crooked | necked | realistic |
|---|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 4/5 | 5/5 | 5/5 | 5/5 | 4/5 | 4/5 | 4/5 |
| Scripted expert, no force feedback | 4/5 | 4/5 | 4/5 | 5/5 | 3/5 | 4/5 | 4/5 |
| Scripted expert, force feedback, estimated geometry | 2/5 | 3/5 | 1/5 | 1/5 | 0/5 | 1/5 | 3/5 |
| Scripted expert, no force feedback, estimated geometry | 1/5 | 2/5 | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 |

By cucumber class (mean coverage):

| Method | pickling | persian | slicing | english | crooked | necked | realistic |
|---|---|---|---|---|---|---|---|
| Scripted expert, force feedback | 99 % | 97 % | 98 % | 98 % | 90 % | 96 % | 97 % |
| Scripted expert, no force feedback | 100 % | 97 % | 97 % | 99 % | 96 % | 97 % | 97 % |
| Scripted expert, force feedback, estimated geometry | 89 % | 92 % | 81 % | 84 % | 40 % | 77 % | 81 % |
| Scripted expert, no force feedback, estimated geometry | 90 % | 91 % | 82 % | 84 % | 40 % | 75 % | 80 % |

Paired comparisons on the same cucumbers (exact McNemar test on success):

- expert_force vs expert_noforce, 35 cucumbers: expert_force wins 3, expert_noforce wins 0, p = 0.25.
- expert_force_nominal vs expert_noforce_nominal, 35 cucumbers: expert_force_nominal wins 7, expert_noforce_nominal wins 0, p = 0.0156.

## Failed first attempt: fine-tuning RDP stage 2 on v1 + v2

The v1 RDP stage 2 was first fine-tuned for 8 epochs on v1 + v2, keeping the v1 tokenizer (stage 1). On 13 v2 test cucumbers it succeeded 0 times and touched the cucumber in 0 of them: the peeler moved and turned far from the cucumber. Most likely cause: the fine-tune refitted the action normalizer to v1 + v2, whose relative motion range in the tool frame is much wider than in v1, while the tokenizer, fixed after stage 1, had learned the v1-normalized actions. RDP's recipe trains both stages on the same data, so the v2 RDP above has both stages trained from scratch on v1 + v2, like the DP baseline.

Diagnostic episodes (`scripts/dev/diag_episode.py`):

- `runs/rdp_ldp/checkpoints/latest.ckpt` on pickling seed 10000: coverage 74 %, blade contact in 1832 control steps.
- `data/big/runs/rdp_ldp_v2_finetune_failed/checkpoints/latest.ckpt` on standard seed 9100: coverage 0 %, blade contact in 0 control steps.

