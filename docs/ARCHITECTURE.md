# Architecture: how the pieces fit

```
                         +-------------------------------+
                         |  Orchestrator (scripted)      |   peel/expert.py: Orchestrator
                         |  reads the skin map, picks the|   (sim: ground-truth map;
                         |  next side, rolls the cucumber|    real: colour segmentation)
                         +---------------+---------------+
                                         | "peel this side now"
                         +---------------v---------------+
  GELLO leader arms  ->  |  Stroke skill                 |   learned: RDP (peel/policy_runner.py)
  (human demos) or       |  approach, touch, press,      |   scripted: PeelExpert (makes the demos)
  scripted expert        |  slide along, lift            |   baseline: Diffusion Policy (vision only)
                         +---------------+---------------+
                                         | TCP targets, 24 Hz
                         +---------------v---------------+
                         |  Cartesian impedance, 480 Hz  |   peel/env.py: Arm.impedance_torque
                         |  (real: libfranka/franky)     |   (real: peel/real_robot.py)
                         +---------------+---------------+
                                         | joint torques
                         +---------------v---------------+
                         |  2 x Franka Emika Panda       |   MuJoCo (peel/scene.py) or real robots
                         +-------------------------------+
```

## Why three layers

- The learning (and the force sensing) only matters while the blade touches the skin. That is the stroke skill.
- Choosing the next side and rolling the cucumber is geometry. A short, checkable program does it better than a
  network, and it is the same in simulation and on the real robot.
- RDP's own peeling task and the Berkeley knife-peeling paper also learn single strokes, not the whole peel.
- The recordings contain whole episodes with both arms, so the same data can train an end-to-end RDP
  (`--kind e2e`, 20-D action) to test whether the orchestrator is needed.

## Inside RDP (Reactive Diffusion Policy, RSS 2025)

- **Stage 1, Asymmetric Tokenizer (AT):** compresses a 32-step (1.33 s) action chunk into a small latent and learns
  a decoder that rebuilds the actions while reading the force history. Config: `rdp_configs/task/sim_stroke_wrench_at.yaml`.
- **Stage 2, Latent Diffusion Policy (LDP):** from the two camera images, the TCP pose and the wrench of the last
  two observation frames, it generates the latent chunk. Config: `rdp_configs/task/sim_stroke_wrench_ldp.yaml`.
- **At run time:** the LDP runs every 4 control steps (6 Hz); the AT decoder runs every step (24 Hz) with the
  newest wrench readings, so the arm reacts to force within one control step even though the planner is slow.
- Changes for this PC's 4 GB GPU: UNet widths 256/512/1024 instead of 512/1024/2048, images 96 x 128 instead of
  240 x 320. Everything else follows RDP's wrench-peeling recipe.

## Code map

| Path | What it does |
|---|---|
| `peel/scene.py` | builds the MJCF: two Pandas from Menagerie, cucumber, peeler, cameras, F/T sites |
| `peel/shape.py` | cucumber shape classes (size, taper, bend), centre line, projection onto the surface |
| `peel/skin.py` | the skin removal rule (cut, skid, gouge) and coverage metrics |
| `peel/env.py` | controller, F/T reading, contact to skin, RDP-format observations, GELLO robot interface |
| `peel/expert.py` | scripted expert, virtual GELLO operator, orchestrator |
| `peel/recorder.py` | episode recording and conversion to RDP's `replay_buffer.zarr` |
| `peel/policy_runner.py` | runs RDP / DP checkpoints in the simulation like RDP's real-robot runner |
| `peel/viz.py`, `peel/rollout.py` | videos and the shared rollout loop |
| `peel/real_robot.py`, `scripts/run_real.py` | real-robot interface for two Pandas (see REAL_ROBOT.md) |
| `scripts/collect_demos.py` | parallel demo collection |
| `scripts/make_rdp_dataset.py` | builds the stroke and end-to-end RDP datasets |
| `scripts/train_rdp.sh` | trains AT, LDP and the DP baseline with the RDP repo |
| `scripts/eval_policies.py`, `scripts/make_results.py` | held-out evaluation by shape class and the results tables |
| `scripts/teleop_gello_sim.py` | real GELLO arms drive the simulation (or replay a recorded stream) |
| `third_party/rdp`, `third_party/gello` | the official RDP and GELLO code, unmodified (RDP's ROS imports are satisfied by `rdp_shims/`) |
