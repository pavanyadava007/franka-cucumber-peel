# How the dataset is created

Short version: **a scripted expert peels simulated cucumbers through the GELLO joint path, and every control step
is recorded in RDP's own data layout.** No human demos and no public dataset are mixed into training.

## 1. Who makes the demos

`peel/expert.py` plays the part of a person holding two GELLO leader arms:

- It plans blade poses along the real (curved, tapered) surface of the cucumber.
- It presses until the wrist force sensor reads its target force (8 to 13 N, drawn per episode), then keeps that
  force during the stroke with a slow integrator on a smoothed force reading, the way an operator watching a force
  bar would. It does not know the skin thresholds.
- It strokes from the gripper end out to the tip at 6 to 11 cm/s with a 10 to 20 degree rake.
- Between strokes the orchestrator measures how wide the last strip was and rolls the cucumber by 80 % of it, then
  fills any gaps at the end.
- Its pose targets are turned into **joint angles** by inverse kinematics, rounded to the GELLO Dynamixel encoder
  resolution (4096 steps per turn), given a little low-pass hand tremor (0.4 mm), and sent through
  `env.command_joint_state()`, which is GELLO's robot interface.

With real GELLO arms, `scripts/teleop_gello_sim.py` sends the human's joint angles through the very same call and
records with the same recorder. `--replay` checks this path without hardware: replaying a recorded command stream
reproduces the recorded episode exactly (seed 2002: coverage 94.6 % both times, peak force 34.13 N both times).

## 2. What is recorded (every 1/24 s)

One `.npz` per episode in `data/raw/`, with RDP key names:

| Key | Shape | Meaning |
|---|---|---|
| `left_robot_tcp_pose`, `right_robot_tcp_pose` | (N, 9) | TCP position + 6D rotation (first two columns of R, RDP convention) |
| `*_tcp_vel` | (N, 6) | TCP twist |
| `*_tcp_wrench` | (N, 6) | external wrench on the tool / cucumber, base frame, tool weight removed |
| `*_gripper_width`, `*_gripper_force` | (N, 1) | gripper state |
| `*_joint_pos` | (N, 7) | joint angles |
| `external_img`, `right_wrist_img` | (N, 96, 128, 3) | camera images, uint8 |
| `action`, `target` | (N, 20) | commanded TCP targets of both arms + gripper widths (RDP's 20-D bimanual layout) |
| `gello_cmd` | (N, 16) | the GELLO joint command that produced the step |
| `phase`, `stroke_id` | (N, 1) | roll / approach / descend / stroke / lift / return, and stroke number |
| `coverage`, `gouge`, `blade_fn`, `cucumber_roll` | (N, 1) | ground truth for analysis (not given to the policies) |

`collect_log.jsonl` has one line per episode with the cucumber shape, physics parameters and outcome.

## 3. Turning it into RDP training data

`scripts/make_rdp_dataset.py` writes `replay_buffer.zarr` with `data/<key>` and `meta/episode_ends`, the layout
RDP's `ReplayBuffer` reads. Only successful demos are used (coverage >= 90 %, gouge <= 3 %).

- `--kind stroke`: one clip per stroke (approach, descend, stroke, lift), peeler arm only, 10-D action
  `[x, y, z, rot6d, gripper]`. This trains the stroke skill used under the orchestrator.
- `--kind e2e`: whole episodes, both arms, 20-D action. This is for the end-to-end variant.

## 4. Public datasets (checked, not used for training)

See DATASET_AUDIT.md. Of 12 public "cucumber peeling" sources, 9 really show cucumber peeling when you look at the
frames; `siyitian/peel_cucumber` shows a robot placing a potato chip despite its name, ForceMimic is zucchini and
the Berkeley knife data could not be checked without downloading about 29 GB. None of them is recorded on a
Franka, so they are useful as references (force levels, formats) and for possible cross-embodiment pretraining,
not as Franka demos.
