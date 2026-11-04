# From simulation to two real Pandas

This guide takes the simulated bimanual peeler to two real Franka Emika Panda arms (FER).
It covers the hardware, the cell, calibration, GELLO teleoperation, demo recording, training and evaluation.

**Status, read this first.**
The real-robot code has not run on a robot.
There was no robot, no GELLO, no camera and no force sensor on the development machine.
What was checked, and how, is listed in [What is verified and what is not](#what-is-verified-and-what-is-not).
Every hardware command in this guide starts as a dry run.
In a dry run the arms are connected and read, but no motion is sent.
Do the first moving runs with two people: one at the enable devices, one at the keyboard.

Code for this guide:

| File | Role |
|---|---|
| `peel/real_robot.py` | `RealBimanualFranka`: the `PeelEnv` interface on real arms. Backends `mock` and `franky`. Safety layer, cameras, Bota sensor hook, calibration helpers. |
| `scripts/run_real.py` | CLI: `status`, `teleop-gello`, `policy`, `scan`. |
| `tests/test_real_mock.py` | Mock tests: interface, observation layout, GELLO and policy paths, safety trips, a franky API contract test. |

## 1. Control stack and why it is franky

The sim sends 24 Hz TCP targets to a Cartesian impedance controller (`peel/env.py`).
The real robots must do the same: one torque-level Cartesian impedance controller per arm, a target that can be updated while it runs, and an external wrench readout.

We use **franky** ([GitHub](https://github.com/TimSchneider42/franky), pip package `franky-control`).

- The Panda needs libfranka older than 0.10, and 0.9.2 is the last libfranka for it ([panda-python on PyPI](https://pypi.org/project/panda-python/)). Franka no longer supports the Panda in current libfranka and franka_ros2 ([arXiv 2608.19740](https://arxiv.org/html/2608.19740)).
- franky 2.0.0 is on [PyPI](https://pypi.org/project/franky-control/) and has wheels built against libfranka 0.9.2 for CPython 3.9 to 3.14 ([wheel index](https://timschneider42.github.io/franky/whl/libfranka-0.9.2/franky-control/)). The project venv is Python 3.10, so the policy and the robot client run in one environment.
- `CartesianImpedanceTrackingMotion` keeps one Cartesian impedance controller running and reads the latest reference every control cycle. It has per-axis stiffness and damping, translational and rotational error clipping, a nullspace posture task, a torque-rate limit (`max_delta_tau`) and force reactions ([impedance guide](https://timschneider42.github.io/franky/guide/impedance-control.html)). The error clip is the same idea as the sim's 5 cm / 0.4 rad clip.
- The external wrench is `RobotState.O_F_ext_hat_K`.

Installation (on the robot PC):

```bash
.venv/bin/pip install franky-control --extra-index-url https://timschneider42.github.io/franky/whl/libfranka-0.9.2/
```

franky also has an index per robot server version ("franky supports Robot Server versions 2 through 10", [installation page](https://timschneider42.github.io/franky/getting-started/installation.html)).
On 2026-11-03 the index for server version 5 served the same libfranka 0.9.2 wheels.
If the robot and libfranka do not match, libfranka refuses to connect (franky raises `IncompatibleVersionException`).
franky asks for a PREEMPT_RT kernel and a `realtime` user group with raised `rtprio` and `memlock` limits (same installation page).
Set `"realtime": false` in the arm config only for tests without a real-time kernel.

**Alternative: Polymetis** ([facebookresearch/fairo](https://github.com/facebookresearch/fairo)).
GELLO's Panda driver (`third_party/gello/gello/robots/panda.py`) and RDP's Franka server (`third_party/rdp/reactive_diffusion_policy/real_world/robot/franka_server.py`) both use it.
RDP calls `start_cartesian_impedance(Kx, Kxd)` and then `update_desired_ee_pose(...)` in a loop, so it does support a runtime target.
We did not pick it for two reasons.
Its client needs Python 3.8 from conda (`third_party/rdp/docs/franka_setup_instructions.md`), while this project runs on Python 3.10.
RDP users report a clash between Polymetis on Python 3.8 and ROS 2 Humble on Python 3.10 ([RDP issue #31](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/31)).
We did not check whether Polymetis can clip the pose error like the sim does.
Other Panda options from the research notes: ROS 1 `serl_franka_controllers` ([GitHub](https://github.com/rail-berkeley/serl_franka_controllers)) and the community ROS 2 Jazzy ports for the Panda ([arXiv 2608.19740](https://arxiv.org/html/2608.19740)).
CRISP on franka_ros2 ([GitHub](https://github.com/utiasDSL/crisp_controllers)) is the better choice for an FR3, but franka_ros2 does not support the Panda.

How the code runs it: each arm gets its own worker process (`_franky_worker_main`).
The worker owns the franky `Robot` and `Gripper` for one IP.
It receives the 24 Hz targets over a pipe, interpolates them linearly at 250 Hz and calls `set_reference`.
Two processes mean two independent libfranka control threads: if one arm faults, the other worker keeps running.
RDP's issue tracker reports jitter on non-Flexiv arms unless a high-rate interpolator runs at 1000 Hz or more ([issue #6](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/6)).
The interpolation is there for that reason.
Our worker interpolates at 250 Hz and franky's controller tracks the latest reference at 1 kHz. Whether 250 Hz is smooth enough is untested; raise `worker_hz` in the config if the arms jitter.

## 2. Bill of materials

Prices come from the research notes (`research_notes/Franka cucumber peeling sim to real/franka_hardware_eval.md`, section Q7) and their sources.
They were not re-checked for this guide.
CHF and USD are not converted, because no exchange rate was fetched.

| Item | Qty | Choice | Price | Source |
|---|---|---|---|---|
| Robot arm | 2 | Franka Emika Panda (FER), lab-owned | not priced | none found |
| Robot PC | 1 | PREEMPT_RT Linux PC, one Ethernet port per robot | not priced | |
| GPU for training | 1 | cloud L4 (as in the project plan) | not priced here | |
| GELLO leader arm, Panda version | 2 | built from [gello_mechanical](https://github.com/wuphilipp/gello_mechanical) (Dynamixel XL330 joints, U2D2 adapter) | not priced | |
| Wrist F/T sensor (optional, phase 2) | 1 | Bota SensONE Gen A | CHF 4,882.50 list | [Bota shop](https://shop.botasys.com/shop/category/force-torque-sensors-4) |
| | | Bota SensONE T15 kit for FR3 (FR3 kit, check Panda fit) | USD 8,198.38 | [Vention](https://vention.com/parts/bota-systems-sensone-t15-gen-a-6-axis-ft-sensor-kit-for-franka-fr3-with-gripper-extension-cable-4909) |
| | | ATI Mini45 (alternative) | from USD 6,675.00 | [e-motion supply](https://www.e-motionsupply.com/product_p/9105-tw-mini45-r-5-ip65.htm) |
| Wrist camera | 1 | Intel RealSense D405 | EUR 327.95 to EUR 414.96 incl. VAT | [MyBotShop](https://www.mybotshop.de/Intel-RealSense-Depth-Camera-D405), [Generation Robots](https://www.generationrobots.com/de/403953-intel-realsense-depth-camera-d405.html) |
| External camera | 1 | Intel RealSense D435 or D455 | not priced | |
| Peeler | 2 or more | Y-style peeler, same model for demos and policy runs | not priced | |
| Peeler adapter | 1 | printed or machined flange adapter with handle clamp and camera bracket | not priced | |
| Finger inserts | 1 pair | Franka Hand inserts for the cucumber, optional spikes | not priced | |
| Kitchen scale, calipers | 1 each | for the evaluation | not priced | |
| Cucumbers | many | about 6 strokes per cucumber in the knife paper, so 50 to 100 demos need roughly 8 to 17 cucumbers plus test produce (notes, Q4) | not priced | [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1) |
| Force-feedback leader (optional) | 2 | FACTR instead of GELLO | about USD 1,229.95 in parts (as given in the report) | [FACTR arXiv](https://arxiv.org/html/2502.17432v2) |

The sensor is the expensive line.
Start without it (phase 1, Franka's estimated wrench), as the research notes recommend.
The Panda datasheet gives 0.8 N relative force accuracy and 0.035 N RMS force noise ([datasheet](https://download.franka.de/Datasheet-EN.pdf)).
The scripted sim expert aims for 8 to 13 N (`ExpertParams.sample` in `peel/expert.py`), so the estimate is usable for contact and force regulation but coarse for peel thickness.

## 3. The cell: copy the sim layout

The sim world frame is the frame of the whole real system.
Numbers below come from `peel/scene.py` and `peel/env.py`.

- Both robot bases sit on the table plane (z = 0), 0.5 m apart, side by side, both facing +x.
  Left base at (0, +0.25, 0), right base at (0, -0.25, 0) (`BASE_Y`).
- The left arm holds the cucumber across the front of both robots.
  The left TCP starts at (0.48, 0.06, 0.30) with the hand z axis along -y (`PeelEnv.LEFT_P0`, `LEFT_R0`).
  For the default 23 cm cucumber the cucumber centre is then at about (0.48, -0.03, 0.30), its axis along -y, tip toward the right arm.
- The left arm turns the cucumber about its own axis with joint 7 only.
  Joint 7 starts 0.08 rad above its lower limit, which gives 5.6 rad (about 320 deg) of roll.
- The right arm carries the peeler.
  It parks 6 cm above the gripper end of the peel zone and strokes from the gripper end out to the tip, edge first, blade dipped 15 deg into the skin (`RAKE0`).
- External camera: at (1.0, -0.32, 0.62), looking at (0.45, -0.06, 0.30), vertical field of view 40 deg.
- Wrist camera: on a side bracket of the peeler adapter, about (-0.116, -0.053, 0.103) m from the right flange in the flange frame, looking at the skin just ahead of the blade, vertical field of view 62 deg.
  The knife-peeling paper found the camera that sees the contact just before the blade the more useful one ([arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)).
  The D405 works best at 7 to 50 cm ([DigiKey](https://www.digikey.de/de/product-highlight/i/intel-realsense/d405-depth-camera)).

Keep the external camera fixed between demo recording and policy runs.
ForceMimic kept its external camera at the same pose for data collection and deployment ([arXiv 2410.07554](https://arxiv.org/html/2410.07554)).

Network: give each Panda its own FCI address and put both into the config (`left.ip`, `right.ip`).
RDP's Franka guide connects the real-time PC directly to the robot by Ethernet (`third_party/rdp/docs/franka_setup_instructions.md`).
With two robots that means two ports on the robot PC.

## 4. The peeler mount (right arm)

Mount the peeler rigidly. Do not hold it in the gripper.
The research notes found that published peeling systems used rigid mounts, and one FR3 study could not make a grasped peeler work (report, section "A rigid peeler mount").

Stack on the right flange: flange, then the F/T sensor (optional), then the adapter, then a clamp around the peeler handle.
The Bota SensONE uses an ISO 9409-1-50-4M6 flange ([Bota SensONE](https://botasys.com/force-torque-sensors/sensone/)); check it against your Panda flange before ordering.
Remove the Franka Hand from the right arm, or keep it and design the adapter around it.
The code assumes no right gripper (`right.has_gripper = false`) and reports the sim's 18 mm as the right gripper width, so the observation matches the training data.

**Blade edge = TCP.** The sim's design transform from the flange to the blade TCP (`default_F_T_tcp("right")`, checked against MuJoCo in the tests) is:

```
F_T_tcp (right) = [[ 0.6409  0.7071  0.2988  0.    ]
                   [-0.6409  0.7071 -0.2988  0.    ]
                   [-0.4226  0.      0.9063  0.205 ]
                   [ 0.      0.      0.      1.    ]]
```

The blade edge centre sits 0.205 m out along the flange z axis.
The rotation is the Franka Hand's -45 deg mounting about the flange z, then the sim's peeler tilt (`peel/scene.py`, `_peeler_xml`).
Design the adapter to hit this transform. Then measure what you built:

1. Translation: pivot calibration. Fix a sharp point in the cell. Touch it with the blade edge centre from at least 4 clearly different orientations (guiding mode). Record the flange poses and run `peel.real_robot.pivot_calibration(poses)`. It returns the tip in the flange frame and an RMS residual.
2. Rotation: take it from the adapter CAD. A pivot calibration cannot see it.
3. Put the 4x4 into the config as `right.F_T_tcp`.

Where the offset goes: the worker reads `F_T_NE` (the end effector configured in Desk) and calls franky `set_ee(NE_T_EE)` so that the robot's EE frame becomes `F_T_tcp`.
Enter the adapter's mass, centre of mass and inertia in Desk as the end-effector load, so the external wrench estimate is not biased by the tool.
For the left arm keep Desk's Franka Hand setting. The sim's left TCP is the Franka Hand fingertip centre, 0.1034 m along the hand z axis.

Printed parts that touch food are not automatically food-safe ([Creality blog, secondary](https://www.creality.com/blog/is-3d-printed-pla-food-safe)).
Treat peeled cucumbers as lab waste, not food.

## 5. Holding the cucumber (left arm)

The left Franka Hand grips the gripped end of the cucumber, like the sim (the cucumber is rigid in the hand there).
The dual-arm layout, with one arm holding and one arm peeling, is the ForceMimic setup ([arXiv 2410.07554](https://arxiv.org/html/2410.07554)).

- Measure each cucumber with calipers: length, diameter at the gripped end, diameter at the tip. Pass them as `--cucumber L D_GRIP D_TIP` (metres). `CucumberFixture` turns them into the sim's tapered shape.
- Push the gripped end against a stop on the finger inserts so it always sits at the same depth. The sim puts the gripped end at 0.080 m along the hand z axis (`--grasp-depth`).
- Optional spiked inserts are meant to stop the cucumber from slipping or turning in the fingers (our design idea, untested). MORPHeus held produce on a spiked board for peeling with a Panda ([arXiv 2404.06570](https://arxiv.org/html/2404.06570v2)).
- The grasp force default is 20 N (`left.grasp_force`). That is our starting value, not a measured one. Tune it on real cucumbers.
- Weigh the cucumber and set `left.load_mass` (and `load_com` in the flange frame) so the left wrench estimate does not carry its weight.

The real system has no perception yet.
The cucumber pose comes from the measured shape and the nominal grasp (`CucumberFixture.hand_T_cucumber`).
The skin map that the orchestrator uses to choose the next roll is an estimate: the sim's own skin rule (`peel/skin.py`) fed with the measured wrench and blade pose.
It is not a measurement. A camera-based skin map is the open TODO in `CucumberFixture`.

## 6. Calibration

Do these in order. Use `python scripts/run_real.py status --backend franky --config <cfg>.json` after each step. It prints joint states, world TCP poses, wrenches, camera frames and the distance between the numpy FK and the robot's own TCP pose.

Write a config template first:

```bash
.venv/bin/python scripts/run_real.py status --write-config configs/real_cell.json
```

**6.1 Base frames.** Mount the bases as in section 3. Then measure the right base relative to the left one:
touch at least 4 marked points on the table with both TCPs, record the TCP positions in each robot's base frame, and run `kabsch(P_right_base, Q_left_base)`. That gives `left_T_right`.
Set `left.world_T_base` to the sim value (translation (0, 0.25, 0)) and `right.world_T_base = left.world_T_base @ left_T_right`.

**6.2 TCPs.** Section 4 for the blade. For the left hand, `status` should report a numpy-FK vs robot TCP distance near zero. A large value means `F_T_tcp`, the Desk end-effector setting or the DH model disagree.
The numpy FK uses Franka's DH table ([robot specifications](https://frankarobotics.github.io/docs/robot_specifications.html), listed there for FR3) and matches the sim's MuJoCo Panda (test `test_numpy_fk_matches_mujoco_and_tool_transforms`). That it matches your Panda's internal kinematics is not verified.

**6.3 Cameras.** Find the serials with `rs-enumerate-devices` and put them into `cameras.external_img.serial` and `cameras.right_wrist_img.serial`.
The default stream is 640x480 RGB at 30 fps, which downsamples to the sim's 96x128 by an exact 5x5 block mean.
Check that your camera supports that colour profile.
For a sim-trained policy the camera must see what the sim camera sees.
Put the external camera at the sim pose from section 3, compare a real frame with a sim frame (`PeelEnv.render("external")`), and use `crop` in the camera config to match the field of view.
For the coverage measurement (section 10) you also need the external camera's pose relative to the left hand. Use any standard hand-eye calibration with a marker held in the left gripper. That step is not implemented here.

**6.4 Force bias.**
Franka estimate: with the loads set in Desk and in the config, hold 5 or more different static poses and read the wrench in `status`. It should stay close to zero. If it does not, the load model is wrong.
Bota sensor: install `bota-driver` ([PyPI](https://pypi.org/project/bota-driver/), [driver docs](https://code.botasys.com/en/gen_a/layer1/driver/driver_py.html)).
Record raw readings in at least 6 clearly different static orientations and run `fit_ft_gravity(raw, R_base_sensor)`.
It fits the bias, the tool mass and centre of mass. A negative fitted mass means the sensor reports the opposite sign convention, and the code then flips it.
Put the result into `right.ft_sensor = {"type": "bota", "config_path": ..., "F_T_sensor": ..., "calib": ...}`.
The Bota hook (`BotaFTSensor`) follows the driver docs (`configure`, `tare`, `activate`, `read_frame`) but was never run with a sensor.

**Sign check, both sources.** The sim reports the wrench of the environment on the tool, in world axes.
Your finger is the environment. Press down on the blade with a finger, and the right `tcp_wrench` z force must go negative. Lift the blade from below, and it must go positive.
In the sim, the cucumber pushes the blade up during a stroke, so peeling shows a positive force along the surface normal.
libfranka's `O_F_ext_hat_K` uses the opposite sign ("Forces applied by the robot to the environment are positive", libfranka `RobotState` docstring shipped with franky 2.0.0), so `franka_wrench_to_sim()` negates it and moves the torque to the flange.
RDP's own Franka server once put joint torques into the wrench field ([issue #28](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/28)). Check this step on the robot, do not trust the code.

## 7. GELLO for the Panda

Build two Panda GELLO leaders ([gello_mechanical](https://github.com/wuphilipp/gello_mechanical)).
On WSL2 attach each U2D2 to Linux first (`usbipd list`, then `usbipd attach --wsl --busid <id>` in an admin PowerShell).

**Offsets.** Put each leader in GELLO's Panda calibration pose and run GELLO's script once per arm:

```bash
.venv/bin/python third_party/gello/scripts/gello_get_offset.py \
    --start-joints 0 0 0 -1.57 0 1.57 0 --joint-signs 1 -1 1 1 1 -1 1 \
    --port /dev/serial/by-id/<adapter>
```

It prints the joint offsets (multiples of pi/2) and the gripper open and closed angles.
Pass them as `--left-offsets`, `--right-offsets` and `--gripper ID OPEN CLOSED`.
The default `--gripper 8 195 152` is GELLO's example entry, not your leader.

**Joint signs: the GELLO docs disagree.** At commit 204f53a:

| Place | Panda signs |
|---|---|
| `third_party/gello/README.md`, "Franka FER (Panda)" command example | `1 1 1 1 1 -1 1` |
| same README, "Joint Signs Reference" | `1 -1 1 1 1 -1 1` |
| `gello/agents/gello_agent.py` `PORT_CONFIG_MAP`, the 7-joint entry (commented "xArm", while the reference lists xArm as all ones) | `1 -1 1 1 1 -1 1` |
| `ros2/README.md` (Franka GELLO for FR3, a different assembly) | `1 -1 1 -1 1 1 1` |

`scripts/run_real.py` defaults to `1 -1 1 1 1 -1 1`.
Check every joint before trusting it, and re-run `gello_get_offset.py` after changing a sign:

1. In the sim first: `scripts/teleop_gello_sim.py` with the viewer. Move one leader joint at a time and watch the follower.
2. On the robots, dry run: `run_real.py teleop-gello --backend franky ...` without `--execute`. The robots do not move. Before the start, the script prints leader minus follower angles per joint. Move one leader joint and check that the matching number changes, in the direction the follower joint would move.
3. Flip the sign of any joint that runs backwards.

Teleop only starts when every leader joint is within 0.8 rad of the follower, the same check as GELLO's `experiments/run_env.py`.

**How GELLO drives the real arms.** As in the sim, leader joints go through forward kinematics to a TCP target, so GELLO and the policy use the same impedance controller.
One difference: the sim also uses the leader joints as the nullspace posture target.
franky 2.0.0 fixes the posture target when the controller starts, so on the robot the elbow is pulled toward the start posture instead.

## 8. Recording human demos

Recordings use the sim's format (`peel/recorder.py`: same keys, 20-D action, `gello_cmd`), plus a `t_wall` array with the real loop time.

```bash
# 1. dry run: robots read but do not move
.venv/bin/python scripts/run_real.py teleop-gello --backend franky --config configs/real_cell.json \
    --left-port ... --right-port ... --left-offsets ... --right-offsets ... --cucumber 0.22 0.038 0.041
# 2. the real thing
.venv/bin/python scripts/run_real.py teleop-gello ... --execute --record --label-strokes --out data_real/teleop
```

- One episode per cucumber: park, approach, stroke, lift, return, roll the cucumber, repeat. Stop with Ctrl+C.
- The terminal shows the blade force estimate. Aim for the sim expert's 8 to 13 N. Live force feedback made demos better in RDP's study, and training on demos without feedback dropped RDP's score by more than 30 % ([RDP arXiv](https://arxiv.org/html/2503.02881v3)). A proper force bar for the real robot is a TODO.
- `--label-strokes` marks contact segments (blade force above 1 N) as strokes, so `make_rdp_zarr(kind="stroke")` can cut clips. Check a few by hand.
- At the end you are asked whether the demo was good. Only good demos have `meta_success = True`, and `make_rdp_zarr` uses only those by default.
- How many: RDP used 60 peeling demos. The knife-peeling paper reached 100 % with 50 single-stroke cucumber demos from about 8 cucumbers ([RDP](https://arxiv.org/html/2503.02881v3), [knife paper](https://arxiv.org/html/2603.03280v1)).

## 9. Training on real demos

Convert real episodes like sim episodes:

```python
from pathlib import Path
from peel.recorder import make_rdp_zarr

real = sorted(Path("data_real/teleop").glob("ep_*.npz"))
make_rdp_zarr(real, Path("data_real/rdp/peel_stroke_zarr"), kind="stroke")
```

Then train with the project script: `DATA=$PWD/data_real/rdp/peel_stroke_zarr bash scripts/train_rdp.sh all`.
Three variants, all untried on real data:

1. **Real only.** The baseline. It is a valid result whatever simulation does.
2. **Co-training.** Build one zarr from sim and real raw episodes: `make_rdp_zarr(sim_files + real_files * k, ...)`. Each listed file is appended once, so listing the real files k times up-weights them. Remember that sim images are low-poly CPU renders, so the image encoder sees two very different domains.
3. **Fine-tuning a sim checkpoint.** RDP's training resumes from `<run dir>/checkpoints/latest.ckpt` when `training.resume` is true (`train_diffusion_unet_image_workspace.py`, lines 129 to 134), and it recomputes the normalizer from the new dataset (lines 142 to 160). Copy the sim run's `latest.ckpt` into a new run directory and train there on the real zarr. The epoch counter resumes too, so set `training.num_epochs` above the sim run's last epoch. Retrain or fine-tune the asymmetric tokenizer (stage 1) as well, because the real wrench signal differs from the sim's.

## 10. Running the policy

```bash
# scripted stroke with force regulation from the measured wrench: the floor baseline, and a hardware check
.venv/bin/python scripts/run_real.py policy --backend franky --config configs/real_cell.json --policy expert
# learned stroke skill (RDP or DP checkpoint) under the orchestrator
.venv/bin/python scripts/run_real.py policy --backend franky --config configs/real_cell.json \
    --ckpt runs/<run>/checkpoints/latest.ckpt --device cuda --record --out data_real/policy
```

Both start as dry runs. Add `--execute` after a clean dry run.
The episode loop is the sim's `run_policy_episode`: the orchestrator rolls the cucumber and parks the peeler, `StrokeSkillRunner` runs each stroke (6 Hz inference, 24 Hz control).
On the robot the orchestrator chooses rolls from the estimated skin map (section 5).

Inference blocks the control loop. `peel/policy_runner.py` runs RDP synchronously, while RDP's own real-robot runner uses threads (see the `policy_runner.py` docstring).
If one slow-policy call takes longer than 0.2 s, the arms hold their pose during it. Longer than 1.0 s stops the controllers.
The CLI runs one timed warm-up inference before the arms start streaming and prints how long it took. If that time is close to the limits, use a faster GPU, fewer diffusion steps, or move inference into a thread first.

Suggested first runs: `--policy expert` without a cucumber, then with one, then the learned skill with `--max-strokes 1`.

## 11. Safety layers

From the policy outward:

1. **Target filter** (`SafetyMonitor.filter_target`): each target is clipped into a per-arm workspace box and limited to 0.25 m/s and 2.0 rad/s of target motion.
   The boxes are the min and max TCP positions of 16 recorded sim episodes plus 10 cm.
   The sim's scripted moves reach 0.52 m/s target speed, so on the robot they run slower. Strokes (0.06 to 0.11 m/s in the expert) are not affected.
2. **Controller**: pose error clipped to 5 cm and 0.4 rad (as in the sim), torque rate limited (`max_delta_tau`).
3. **Force abort**: a franky reaction stops the controller inside the 1 kHz loop when |F| > 30 N. The main loop also aborts both arms when |F| > 30 N on either arm.
   In 16 recorded sim episodes the wrench peaked at 26.7 N (right) and 26.8 N (left). 30 N leaves little margin. Expect some aborts and look at them before raising the limit.
4. **Measured state**: abort if a TCP leaves its box by more than 3 cm, moves faster than 0.8 m/s (sim peak 0.49 m/s), or the robot state is older than 0.25 s. In a dry run these are logged instead, because the arms are not ours to move.
5. **Heartbeat**: if an arm gets no new target for 0.2 s it holds its measured pose; after 1.0 s its worker stops the controller. The main loop aborts if two control steps are more than 1.0 s apart.
6. **libfranka collision reflex**: franky's defaults (20 Nm, 30 N) are passed to `Robot(...)`. Tune with `torque_threshold` and `force_threshold`.
7. **People**: enable device in hand for each arm, a blade guard when not peeling, and cut-resistant gloves for anyone loading cucumbers (research notes, Q6). Wipe off cucumber juice and use a drip tray. The notes give IP40 for the FR3 arm from a secondary source; check your Panda's rating.

After any abort the env refuses to step until `reset()`. `reset()` runs libfranka's automatic error recovery on both arms; if that fails, clear the error in Desk.

## 12. Evaluation protocol

Run **20 trials per condition**. ForceMimic used 20 trials for its main methods ([arXiv 2410.07554](https://arxiv.org/html/2410.07554)). Report success rates with a confidence interval (for example Wilson).

Conditions (from the report): nominal cucumber and grasp; diameter and curvature bins measured with calipers; a shifted grasp; a push after contact, as in RDP's perturbation tests ([RDP](https://arxiv.org/html/2503.02881v3)); zucchini as out-of-distribution produce ([knife paper](https://arxiv.org/html/2603.03280v1)).
Methods: `--policy expert` (scripted floor), vision-only DP, RDP on the Franka estimate, RDP on the Bota sensor, and the sim-trained variants from section 9.

Per trial, measure:

1. **Coverage by camera segmentation.** After the trial run `run_real.py scan --backend franky ... --execute`. It parks the peeler, rolls the cucumber through its joint 7 range in `--scan-views` steps and saves one full-resolution external frame per view. Segment skin (dark green) against flesh (pale) in each frame. The knife paper used SAM2 masks online ([arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)). Hand-label a few frames to set and check the segmentation. Coverage = peeled area / visible cucumber area over all views. Joint 7 cannot turn a full circle (about 320 deg), so record what the views miss.
2. **Weighing.** Weigh the cucumber before and after, and weigh the collected peel. Removed mass = before minus after. Peel mass per peeled area is a thickness proxy (report, evaluation section). The difference between removed mass and collected peel is juice and lost debris.
3. **Gouges**: count spots where the blade dug into the flesh, by eye or in the segmentation.
4. Strokes, time, mean and peak blade force, number of safety stops.

The research notes found no published rubric for segmentation coverage or peel weighing (Q5). Write the protocol down before the first trial and keep it fixed.

## 13. Known sim-to-real gaps

- **Rule-based skin.** The skin is a grid of cells removed by a threshold rule on force, rake angle and blade motion (`peel/skin.py`). The thresholds are assumptions, randomised per episode.
- **No measured cucumber mechanics.** The research found no published cucumber skin mechanics, so nothing in the sim is calibrated to a real cucumber. Calibration strokes with force logs and peel weights are the first real-robot experiment worth doing (report, simulation section).
- **Images.** Policy images are CPU-rendered from low-poly collision meshes at 96x128 (`PeelEnv.render`). They look nothing like RealSense frames.
- **Scripted demos.** No human recorded the sim demos. A scripted expert generated them through the GELLO joint path (`peel/expert.py`). Human demos will have different timing, forces and mistakes.
- **Rigid grasp in the sim.** The cucumber is welded to the left hand there. On the robot it can slip or turn in the fingers.
- **Force signal.** The sim adds white noise to a clean wrench. Franka's estimate depends on the load model, and the research notes expect it to suffer during fast motion (their inference, not a measurement). RDP notes that estimated wrenches are noisier than a wrist sensor such as an ATI Mini45 ([RDP](https://arxiv.org/html/2503.02881v3)).
- **Control details.** The real arms interpolate 24 Hz targets at 250 Hz, the sim holds each target for one step. The real nullspace posture is fixed at the start pose. The real target speed limit slows scripted moves. Real loops have latency that the sim does not model.
- **Estimated skin map.** On the robot the orchestrator rolls the cucumber based on what the sim's rule predicts happened, not on what happened. A camera skin map would close this gap.

## What is verified and what is not

| Item | How it was checked | Status |
|---|---|---|
| `RealBimanualFranka` on the mock backend: obs keys, shapes and dtypes equal `PeelEnv.obs()`; start pose, cucumber frame, park pose and roll range equal the sim's | `tests/test_real_mock.py` | tested (mock only) |
| GELLO path (`command_joint_state`), recording and `make_rdp_zarr` on mock recordings | tests, manual run | tested (mock only) |
| `StrokeSkillRunner` and the orchestrated episode loop on the real env, with a stub policy | tests | tested (mock only), no trained checkpoint |
| Safety: force abort and latch, workspace clip and abort, speed limits, heartbeat abort, backend watchdog hold, dry-run logging | tests | tested (mock only) |
| Numpy Panda FK vs the sim's MuJoCo model; sim tool transforms; Jacobian | tests | tested |
| libfranka-to-sim wrench conversion, F/T gravity fit, pivot calibration, base registration | tests with synthetic data | tested (math only) |
| franky API used by the arm worker | franky 2.0.0+libfranka.0.9.2 wheel stubs read; the worker run against the real franky package with only `Robot` and `Gripper` faked (`test_franky_worker_contract_with_fake_robot`, skipped when franky is not installed) | API contract checked, no robot |
| franky on a Panda: real-time loop, impedance behaviour, `set_ee`, gripper timing, reaction stop | none | **untested on hardware** |
| Two arms in two processes on one PC | none | **untested on hardware** |
| pyrealsense2 camera code | API names checked against pyrealsense2 2.59 | **untested on hardware** |
| Bota SensONE hook | written from the driver docs and package docstring | **untested on hardware** |
| GELLO Panda signs and offsets | none (no leader arms) | **untested on hardware** |
