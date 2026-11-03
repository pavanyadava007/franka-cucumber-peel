# Real-robot hardware, control, data collection and evaluation stack for Franka FR3/Panda cucumber peeling

Scope note: researched 2026-11-02. Sources are official Franka docs/brochure, GitHub READMEs, arXiv papers (read via full-text passages) and vendor/retailer listings. Important framing finding up front: the three best-known learned peeling papers (ForceMimic, RDP, ForceVLA) all ran on **Flexiv Rizon** arms, not Franka. The only peeler-on-Franka system found is MORPHeus (Panda, Cornell). The best-documented 2026 learned peeling system (Berkeley "How to Peel with a Knife") ran on a **Kinova Gen3** with an ATI Mini45. So a Franka cucumber-peeling setup has to transfer designs across robots.

## Q1. What is the most reliable 2025-2026 open-source control stack for learned impedance/force policies on an FR3 (and what changes on a Panda)?

### Takeaway
For an FR3 in late 2026, the best-evidenced choice is **CRISP (utiasDSL/crisp_controllers) on franka_ros2 (ROS 2 Jazzy) + current libfranka (0.19.0) + PREEMPT_RT kernel**: it runs Cartesian impedance / OSC at 1 kHz, exposes a target-wrench term and runtime-configurable stiffness, was validated on FR3 hardware, and was shown tracking Diffusion Policy at ~30 Hz and SmolVLA at ~10 Hz. On a Panda, vendor software has stopped (libfranka < 0.10 only), so you are on ROS 1 tools (serl_franka_controllers, Deoxys, Polymetis) or community ROS 2 ports (multipanda_ros2, unisa-acg Jazzy port).

### Cited Findings
**FR3 / FCI basics (official)**
- FCI lets an external workstation send real-time commands at 1 kHz via five interfaces: joint torque (with gravity and friction compensation), joint position or velocity, Cartesian pose or velocity; it simultaneously streams at 1 kHz joint data, link-side torque sensor signals, "estimation of externally applied forces", and collision/contact information; the model library gives FK, Jacobians, inertia, Coriolis and gravity - [Franka Research 3 brochure, Feb 2026](https://franka.de/hubfs/Brochure_Franka%20Research%203_February%202026_English.pdf)
- FR3 System Image v5.8 adds a Desk API (programmatically open brakes, activate FCI, clear safety violations) and in-field torque sensor recalibration "to deliver consistent accuracy during the complete lifetime" - [FR3 brochure, Feb 2026](https://franka.de/hubfs/Brochure_Franka%20Research%203_February%202026_English.pdf)
- "FCI now supports asynchronous joint-position updates and full access to the robot's datasheet-level joint limits" - [franka.de FR3 product page](https://franka.de/franka-research-3)
- FR3 headline specs: payload 3 kg, reach 855 mm, repeatability < +/-0.1 mm, 7 integrated torque sensors - [franka.de FR3 product page](https://franka.de/franka-research-3)
- Secondary (review site, not official): FR3 translational stiffness range 10-3,000 N/m, rotational 1-300 Nm/rad, guiding force ~2.5 N, joint torque limits +/-87 Nm (A1-A4) and +/-12 Nm (A5-A7), arm IP40; and "If a paper says it uses a Franka Panda, do not automatically describe that experiment as an FR3 result" - [Anton Robots FR3 review](https://antonrobots.com/franka-research-3-review/)

**libfranka / franka_ros2 versions and real-time**
- libfranka README install examples use **libfranka 0.19.0**; supported Ubuntu 20.04, 22.04, 24.04, 26.04 (amd64); pylibfranka Python bindings exist (3.9-3.12 on 22.04/24.04); a PREEMPT_RT kernel is "recommended for real-time control" - [frankarobotics/libfranka GitHub](https://github.com/frankarobotics/libfranka)
- franka_ros2 "is built upon ROS 2 jazzy"; supports FR3, FR3 Duo and Mobile FR3 Duo; example controllers named include `joint_impedance_example_controller`, `cartesian_velocity_example_controller`, `cartesian_elbow_example_controller`; "A real-time kernel is essential to ensure proper communication and to prevent timeout issues"; recommends Docker Engine over Docker Desktop - [frankarobotics/franka_ros2 GitHub](https://github.com/frankarobotics/franka_ros2)
- Compatibility is keyed to robot system version, e.g. system 5.7.2 "needs a robot version 9 and gripper version 3"; franka_ros2 distributions listed are Jazzy and Humble; no Windows support - [Franka compatibility docs](https://frankarobotics.github.io/docs/compatibility.html)
- Panda/FER: "FER supports libfranka version <0.10 while the FR3 requires version >..." and libfranka 0.9.2 is "the newest version for the Franka Emika Robot" - [panda-python on PyPI](https://pypi.org/project/panda-python/)
- "Vendor software support for the Panda, however, has been discontinued: current releases of franka_ros2 and libfranka target the newer FR3 ... but no longer support the Panda"; Panda external position/Cartesian commands are "prone to vibrations and to protective stops"; the authors port franka_ros2 to Jazzy for the Panda with an extended libfranka that "can run on a stock kernel without a real-time patch" (repo unisa-acg/fer_moveit_config) and cite multipanda_ros2 (Skerlj, Haddadin et al. 2026) - [Keeping the Franka Emika Panda alive, arXiv 2608.19740](https://arxiv.org/html/2608.19740)

**CRISP (recommended for FR3)**
- C++ ros2_control controllers "capable of running at 1 kHz on a joint-torque interface"; users "publish target commands at any frequency" and the controller tracks the latest; Cartesian Impedance (CI) and Operational Space Control (OSC), nullspace control, joint-limit barriers, friction compensation (FR3 parameters provided), Pinocchio dynamics - [CRISP, arXiv 2509.06819](https://arxiv.org/html/2509.06819)
- Target wrench: "it is possible to command a wrench at the end-effector", tau_wrench = J^T F_target, each term of the control law can be switched on/off - [CRISP, arXiv 2509.06819](https://arxiv.org/html/2509.06819)
- "all controller parameters are dynamically configurable via standard ROS2 parameter services", including stiffness, damping, nullspace stiffness, torque rate limits, error limits; safety features: joint torque and rate limits, exponential target filtering, tracking error clipping - [CRISP, arXiv 2509.06819](https://arxiv.org/html/2509.06819)
- Validated on real FR3: steady-state position errors 5.54 mm (OSC), 4.73 mm (CI), 0.81 mm (CI-clipped); Diffusion Policy deployed at ~30 Hz and SmolVLA at ~10 Hz; leader-follower teleop streamed at ~30 Hz with force-torque feedback - [CRISP, arXiv 2509.06819](https://arxiv.org/html/2509.06819)
- README: "all ROS 2 distributions are supported from a single main branch"; crisp_py Python interface; crisp_gym Gymnasium env that can "record trajectories in LeRobotFormat"; IEEE 2026 publication badge - [crisp_controllers GitHub](https://github.com/utiasDSL/crisp_controllers)
- CRISP's own comparison: Cartesian Impedance Controller (CIC), compliant_controllers, serl_franka_controllers "are only available for ROS1 or are specific to a single manipulator type" - [CRISP, arXiv 2509.06819](https://arxiv.org/html/2509.06819)

**Other wrappers**
- serl_franka_controllers: "Cartesian impedance controller with reference limiting for Franka Emika Robot"; limiting the reference "allows a high gain ... without excess force when in contact"; requires ROS Noetic and libfranka/franka_ros >= 0.8.0; compliance via rqt_reconfigure / dynamic_reconfigure from Python - [serl_franka_controllers GitHub](https://github.com/rail-berkeley/serl_franka_controllers)
- HIL-SERL (uses that controller): impedance controller at 1000 Hz accepting 10 Hz policy setpoints; bounding |e| <= Delta bounds spring-damper force to k_p*Delta + 2*k_d*Delta*f; for dynamic tasks the policy outputs a 3-DoF feedforward wrench, executed by a 1 kHz feedforward wrench controller (J^T) - [HIL-SERL, arXiv 2410.21845](https://arxiv.org/html/2410.21845v3)
- franky: Python/C++; Ruckig trajectory control in joint and Cartesian space; torque control incl. built-in Cartesian and joint impedance; motions "can be preempted at any moment"; asks for a real-time kernel; "Full Python access to the libfranka API" (runtime stiffness change and force targets not explicitly documented in README) - [franky GitHub](https://github.com/TimSchneider42/franky)
- Deoxys: "a modular, real-time controller library for Franka Emika Panda arm"; C++ Franka Interface runs on a NUC, Python on desktop; README does not mention FR3 - [deoxys_control GitHub](https://github.com/UT-Austin-RPL/deoxys_control)
- DROID stack: Franka Panda + Robotiq 2F-85, 2x ZED 2 + wrist ZED Mini, Polymetis controller on a NUC, actions recorded at 15 Hz, Meta Quest 2 teleop; replicated 18 times - [DROID, arXiv 2403.12945](https://arxiv.org/html/2403.12945v2)
- Polymetis is described as "a PyTorch-based real-time controller manager" in facebookresearch/fairo - [fairo GitHub](https://github.com/facebookresearch/fairo)
- 2026 alternative: "Plug, Play and Comply" ROS 2 plugin framework whose command interfaces support "motion references, stiffness and damping gains, nullspace targets, desired interaction wrenches, and independent feedforward inputs" (Pinocchio, real-robot curved-surface following) - [arXiv 2607.22483](https://arxiv.org/abs/2607.22483)
- Example of a 2026 FR3 learning pipeline using an admittance controller with residual RL on the reference pose (7 demos, 40/40 OOD trials) - [Residual Compliant Policy, arXiv 2610.14035](https://arxiv.org/abs/2610.14035)

### Inferences
- Variable stiffness + target wrench from a 10-30 Hz learned policy: **CRISP** (both, FR3-validated, ROS 2), **Plug, Play and Comply** (both in the command interface, newer and less battle-tested), **serl/HIL-SERL** (stiffness via dynamic_reconfigure, feedforward wrench variant; ROS 1 Noetic only), **franky** (impedance yes; per-step stiffness/wrench not documented, verify in code). Deoxys and Polymetis are Panda-era stacks with no FR3 evidence found.
- CRISP stiffness is set through ROS 2 parameter services; whether that is intended for per-step (10-30 Hz) stiffness updates from a policy is not stated. For ACP-style learned stiffness, plan to add a stiffness field to the target message or verify parameter-update latency on the real robot.
- HIL-SERL-style reference limiting / CRISP error clipping is the key safety mechanism for a blade task: it caps the force the impedance spring can produce regardless of what the policy outputs.
- If the lab robot is a Panda, the clean path is: system 4.x/libfranka 0.9.2 + multipanda_ros2 or the unisa-acg Jazzy port, or ROS 1 + serl_franka_controllers. Do not mix FR3 instructions with a Panda.

### Gaps
- Could not retrieve the official FR3 libfranka/system-version matrix contents (page links to libfranka-common headers); exact minimum libfranka for a given FR3 system image not confirmed.
- Could not confirm whether Polymetis/fairo is archived or still maintained.
- franka_ros2 README excerpt did not list a Cartesian impedance example controller; full controller list not verified.
- No head-to-head benchmark of CRISP vs franky vs serl on FR3 found.

## Q2. Is Franka's built-in external wrench estimate good enough for peeling force feedback, or is a wrist F/T sensor necessary?

### Takeaway
Peeling forces in published peeler systems are on the order of **~10 N (ForceMimic data ~10 N, good policy ~9 N, bad policy ~20 N with peaks >40 N)**, not "a few newtons". Franka's published Panda spec of **0.8 N relative force accuracy, 0.15 N repeatability, 0.035 N RMS noise** is adequate for contact detection and coarse force regulation, and RDP/ForceVLA showed estimated wrench is usable as a policy input. But every paper that used estimates flagged their noise/model error, and the strongest 2026 peeling result used an ATI Mini45. Recommendation: start with the built-in estimate, add a wrist F/T sensor (Bota SensONE kit for FR3) if you want publication-grade force logs, force-conditioned policies, or thickness-sensitive control.

### Cited Findings
- Panda datasheet sensing: force resolution < 0.05 N, force repeatability 0.15 N, torque resolution 0.02 Nm, force noise (RMS) 0.035 N, relative force accuracy 0.8 N - [Franka datasheet (download.franka.de)](https://download.franka.de/Datasheet-EN.pdf); note copies differ on repeatability (< 0.15 N in [WiredWorkers copy](https://www.wiredworkers.io/wp-content/uploads/2019/12/Panda_FrankaEmika_ENG.pdf) vs < 0.05 N in [ULB Sibiu copy](https://centers.ulbsibiu.ro/incon/wp-content/uploads/Datasheet-Franka-Emika-Panda-Research-collaborative-robot.pdf))
- FR3 provides "Estimation of externally applied forces" at 1 kHz and in-field torque sensor recalibration (v5.8), but the Feb 2026 brochure gives no numeric force accuracy - [FR3 brochure, Feb 2026](https://franka.de/hubfs/Brochure_Franka%20Research%203_February%202026_English.pdf)
- ForceMimic zucchini peeling: Raw DP peeling forces "averaging around 20N and reaching over 40N"; training data "around 10N"; HybridIL "maintained an average interaction force of 9N ... resulting in evenly peeled sections with consistent thickness"; hybrid force control triggered when predicted force "exceeds 6N continuously" - [ForceMimic, arXiv 2410.07554](https://arxiv.org/html/2410.07554)
- RDP: "the estimated TCP force / torque signals have relatively larger noise compared to the force sensor mounted on the robot end effector (e.g., ATI mini 45) due to inaccurate dynamics model"; they still used Flexiv RDK estimated TCP wrench streamed at 120 Hz, downsampled to 24 FPS, and RDP (force) beat DP baselines by > 35% across peeling, wiping, bimanual lifting - [RDP, arXiv 2503.02881](https://arxiv.org/html/2503.02881v3)
- ForceVLA limitation: "ForceVLA currently utilizes estimated external wrench values ... these estimations may not always capture the full precision afforded by direct high-fidelity measurements" - [ForceVLA, arXiv 2505.22159](https://arxiv.org/html/2505.22159)
- Berkeley knife peeling: ATI Mini45 between flange and end-effector at 500 Hz; modality ablation on potato: gRGB+D+F 80% success vs gRGB+D 60% vs F only 0% - [How to Peel with a Knife, arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)
- FACTR relays Franka's external joint torque tau_ext to a leader arm and notes tau_ext "is a readily available measurement in various collaborative robot manipulators, such as the Franka Panda" - [FACTR, arXiv 2502.17432](https://arxiv.org/html/2502.17432)
- General evidence that manufacturer estimates are imperfect: LETS reports manufacturer external-torque estimate joint RMSE 0.54 Nm (free space) and 0.57 Nm (contact) vs 0.17/0.41 Nm for their learned residual (robot model not identified in the passages read) - [LETS, arXiv 2610.14889](https://arxiv.org/html/2610.14889); a DNN reduced force error "by around 57%" vs the manufacturer's model (robot not confirmed in what was retrieved) - [Frontiers in Robotics and AI 2022, PMC9106527](https://pmc.ncbi.nlm.nih.gov/articles/PMC9106527/); FILIC proposes a Jacobian-inversion joint-torque estimator with digital-twin compensation for robots without F/T - [FILIC, arXiv 2509.17053](https://arxiv.org/abs/2509.17053)
- F/T sensors are more accurate than joint-torque estimates because they do not depend on the dynamic/kinematic model, but drift over time - [arXiv 2403.01068](https://arxiv.org/abs/2403.01068)
- Robotiq FT 300-S: +/-300 N, +/-30 Nm, 0.1 N force signal noise, 100 Hz over RS-485, 442 g, IP65, couplings listed for UR/Techman/Omron - [Tavoron listing](https://tavoron.com/product/robotiq-ft-300-s-force-torque-sensor/); manual: 1 s Fz std dev 0.1 N, "relative force measurement are recommended from 1N" - [Robotiq knowledge base](https://blog.robotiq.com/knowledge/robotiq-ft-300-sensor-ft-300-s-sensor-and-copilot-software-general-measurements-5-1736280750218)
- Bota sells a "SensONE T15 Gen A 6-axis F/T Sensor Kit for Franka FR3 with gripper extension cable"; SensONE uses an ISO 9409-1-50-4M6 flange and includes a 6-DoF IMU - [Vention listing](https://vention.com/parts/bota-systems-sensone-t15-gen-a-6-axis-ft-sensor-kit-for-franka-fr3-with-gripper-extension-cable-4909); [Bota SensONE page](https://botasys.com/force-torque-sensors/sensone/)

### Inferences
- With ~9-10 N nominal peel force, a 0.8 N relative accuracy is roughly 8-9% of the signal: fine for "in contact / too hard / lost contact" logic and for a force feature in the policy, marginal for regulating peel thickness, which depends on sub-newton and sub-millimetre differences (RDP reports its corrective actions are sub-millimetre).
- Built-in estimates also depend on correct end-effector load (peeler + adapter mass and CoM) and suffer during fast motion; a wrist F/T measures the tool-produce interaction directly and gives a sim-to-real-comparable signal.
- Practical plan: phase 1 use O_F_ext_hat_K (FCI external wrench) with a correct load model and record it; phase 2 add a Bota SensONE (FR3 kit) between flange and a rigid peeler adapter and compare both signals on the same strokes, which itself is a useful result for a sim-to-real paper.
- Robotiq FT 300-S (0.1 N noise, 100 Hz) is weaker for a 24-30 Hz fast-policy loop than Bota/ATI and needs a Franka adapter plate.

### Gaps
- No FR3-specific numeric force accuracy found in official 2026 material (only Panda datasheet values).
- No paper found that directly measured Franka O_F_ext_hat_K error against an ATI/Bota sensor during tool-contact tasks with numbers in newtons.
- No measured cucumber-specific peeling force found (ForceMimic numbers are zucchini).

## Q3. What fixture and tool-mount designs have worked in published peeling/cutting systems?

### Takeaway
Three holding strategies are published: **second arm holds the vegetable** (ForceMimic; earlier Watanabe and Dong dual-arm peelers), **human holds it mid-air** (RDP, used to test perturbation robustness), and **single arm + assistive cutting board with moving jaw or spikes** (MORPHeus, the only Franka Panda peeler system). Tool mounting is either **grasped by the gripper** (RDP), a **fixed peeler identical to the human data-collection tool** (ForceMimic), or a **custom rigid end-effector mount with F/T plate and two wrist-camera mounts** (Berkeley knife peeling). For a single FR3, the evidence points to a rigid flange-mounted peeler adapter (through an F/T sensor) plus a clamping/indexing fixture, with compliance supplied by the impedance controller rather than the mount.

### Cited Findings
- ForceMimic: "The left robot, equipped with a gripper, was used for rule-based stabilization of the zucchini, while the right arm's fixed peeler identical to the one used in ForceCapture, performed the peeling skill"; robot Flexiv Rizon 4; external L515 RGB-D camera kept at the same pose for data collection and deployment; data collection used the "fixed-tool version of ForceCapture" with the user holding the zucchini in the left gripper; efficiency study peeled "the zucchini on a stand" and re-grasped to rotate it - [ForceMimic, arXiv 2410.07554](https://arxiv.org/html/2410.07554)
- RDP: "The robot needs to grasp the peeler, approach a cucumber held midair by a human hand, then begin peeling"; requires "millimeter-level" precision under different tool grasp locations and cucumber poses; hardware two Flexiv Rizon 4 + Flexiv Grav grippers, single wrist RealSense D435 for single-arm tasks; GelSight Mini and MCTac on fingertips - [RDP, arXiv 2503.02881](https://arxiv.org/html/2503.02881v3)
- ForceVLA: Flexiv Rizon 7-DoF + Dahuan adaptive gripper; third-person RealSense D435 (1280x720, 30 FPS) and wrist RealSense D415 (640x480, 30 FPS) - [ForceVLA, arXiv 2505.22159](https://arxiv.org/html/2505.22159)
- Berkeley knife peeling: Kinova Gen3; ATI Mini45 between tool flange and end-effector; "custom end-effector mount design, including an arm connector, a force-torque sensor plate, two camera mounts, and a knife mount"; two RealSense D405 wrist cameras, one seeing the contact "slightly before" and one "slightly after" the blade; the "before" camera mattered more (less occluded); impedance control at 500 Hz, Python commands at 10 Hz on an Intel NUC - [How to Peel with a Knife, arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)
- MORPHeus (Franka Emika Panda): single arm with an "assistive cutting board" (adaptive board with fixtures); board variants with a moving jaw (slide in, lock, peel, unlock, rotate food, repeat) and a spiked board (food pushed onto spikes); board fixed to the table with suction cups; a 3D-printed peeler holder fixed to the table; peeler instrumented for vision, force and vibration (open hardware); Cartesian impedance controller - [MORPHeus, arXiv 2404.06570](https://arxiv.org/html/2404.06570v2)
- MORPHeus failure analysis: items too large slip from the jaw, small/spherical items evade it, the dual-peg design can occlude the middle section; they recommend larger boards or spiked boards for size variation - [MORPHeus, arXiv 2404.06570](https://arxiv.org/html/2404.06570v2)
- Prior peeler systems by Watanabe et al. and Dong et al. "both used a dual-arm setup: one arm stabilizes the food items, and the other peels" - [MORPHeus related work, arXiv 2404.06570](https://arxiv.org/html/2404.06570v2)
- Dexterous alternative: a hand reorients and holds the vegetable while a separate peeler peels - [Vegetable Peeling: constrained dexterous manipulation, arXiv 2407.07884](https://arxiv.org/abs/2407.07884)
- Bota SensONE has an ISO 9409-1-50-4M6 mounting flange (same family as the FR3 DIN ISO 9409-1-A50 flange listed by the review) - [Bota SensONE](https://botasys.com/force-torque-sensors/sensone/); [Anton Robots FR3 review](https://antonrobots.com/franka-research-3-review/)

### Inferences
- A cucumber is long and cylindrical, so a **between-centres / two-spike fixture with manual or motorised indexing** (rotate 30-45 degrees between strokes) is the natural single-arm generalisation of MORPHeus's spiked board and of the "rotate between strokes" pattern in ForceMimic and ForceVLA. Clamp V-blocks risk occluding the surface the way MORPHeus's pegs did.
- A **rigid flange mount** (flange -> F/T -> printed/aluminium adapter -> peeler handle clamp) removes grasp-pose variation that RDP had to learn around, and keeps the F/T signal clean; RDP-style gripper grasp is better only if tool pick-up is part of the task.
- Keep compliance in the controller (bounded-error impedance, CRISP clipping) not in the mount, so the sim model and the real robot share one compliance parameterisation; this is an inference, not a published comparison.
- A second fixed wrist camera "before the blade" (Berkeley finding) is cheap to replicate with a D405 on the adapter.

### Gaps
- No paper found that compared **swivel-blade vs fixed-blade** peelers or reported which type was used in detail (ForceMimic/RDP/ForceVLA excerpts do not specify).
- No published design with **deliberate passive compliance in the peeler mount** was found.
- ForceVLA cucumber fixture details and ForceMimic adapter CAD dimensions were not in the passages read (ForceMimic says hardware is on forcemimic.github.io).
- No Franka-specific cutting paper with fixture details was retrieved in this session.

## Q4. How should force-rich peeling demonstrations be collected, and how many are needed?

### Takeaway
For peeling, **SpaceMouse teleoperation on a compliant controller was the best published option** (100% usable demos, score 8.5/9, vs VR 20%), and **50 single-stroke demos (~8 cucumbers) gave 100% cucumber success** on a torque-controlled arm. Force/tactile feedback during teleop measurably improves data and policy quality (TactAR AR feedback; FACTR force-feedback leader arm). Handheld force-capture (ForceMimic) is ~3x faster than teleop but has a robot/controller mismatch problem at deployment.

### Cited Findings
- Berkeley knife peeling data-collection comparison on cucumbers (10 trajectories each): Heuristic planner 20% success / score 1.8; VR 20% / 2.1; **SpaceMouse 100% / 8.5 / 46 s**; Kinesthetic teaching 100% / 7.2 / 13.8 s, but replay of kinesthetic data 0% / 1.6 even after compliance tuning; VR suffered from controller holding instability and tracking noise - [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)
- Same paper: demos are single strokes recorded at 10 Hz (joint angles, F/T, two wrist RGB-D); 50 cucumber demos (~8 cucumbers) -> 100% success, score 8.5; 100 apple demos (~17 apples) -> 60%; potato 40 / 100 / 200 demos -> 10% / 60% / 80% (200 potato demos ~33 potatoes); each number over 10 trials; SAM2 used online for knife/object masks - [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)
- RDP / TactAR: Meta Quest 3 AR teleop (~$500) rendering tactile/force as a 3D deformation field; 10-user peeling study (200 trials): feedback raised normalized peel length 0.72 -> 0.91 and stable-contact-force ratio 0.58 -> 0.87; 60 peeling demos; training on 60 demos collected without feedback dropped RDP (force) score by > 30% - [RDP, arXiv 2503.02881](https://arxiv.org/html/2503.02881v3)
- RDP control rates: slow latent diffusion policy 1-2 Hz action chunks (12 FPS actions), fast tokenizer policy at 24 Hz on tactile/force (limited by GelSight 25 FPS) - [RDP, arXiv 2503.02881](https://arxiv.org/html/2503.02881v3)
- ForceMimic / ForceCapture: robot-free handheld force-motion capture; "an operator can peel a zucchini in 5 minutes, while force-feedback teleoperation takes over 13 minutes"; teleop (RH20T setup) took ~3x longer with 3 interruptions; dataset 15 zucchinis, 438 peeling segments, 30,199 action sequences; deployment force mismatch (20 N vs 10 N in data) blamed on controller inconsistency between collection and deployment - [ForceMimic, arXiv 2410.07554](https://arxiv.org/html/2410.07554)
- ForceVLA: Quest 3 VR teleop, ~50 demos per task, 244 trajectories / 140k timesteps across 5 tasks, 5 operators - [ForceVLA, arXiv 2505.22159](https://arxiv.org/html/2505.22159)
- FACTR: GELLO-like 3D-printed servo leader arm with active force feedback from Franka tau_ext, gravity and friction compensation, nullspace redundancy resolution; vs unactuated GELLO-style baseline: 64.7% higher completion, 37.4% less time, 83.3% better ease of use; GELLO relies on springs that cause "non-uniform, configuration-dependent wrenches"; BOM in appendix - [FACTR, arXiv 2502.17432](https://arxiv.org/html/2502.17432)
- CRISP supports leader-follower teleop with force feedback tau_fb = -k_p J^T F_follower - k_d qdot_leader, ~30 Hz, recording directly in LeRobot format - [CRISP, arXiv 2509.06819](https://arxiv.org/html/2509.06819)
- UMI-style tactile capture: FARM modifies the UMI handheld gripper with a GelSight Mini and builds an actuated twin for deployment - [FARM, arXiv 2510.13324](https://arxiv.org/abs/2510.13324)

### Inferences
- For an FR3 + CRISP setup, the lowest-risk first pipeline is SpaceMouse teleop into a clipped Cartesian impedance controller, single-stroke episodes, 50-100 demos, rotating the cucumber between strokes. Add an AR or GUI force bar (TactAR idea) to keep contact force stable, since data stability directly affected RDP performance.
- If a second FR3 or a GELLO/FACTR leader is available, bilateral force-feedback teleop is the next upgrade; kinesthetic teaching is fast but its implicit compliance does not replay.
- Demo budget of cucumbers: ~6 strokes per item in the knife paper, so 50-100 demos is roughly 8-17 cucumbers plus evaluation produce.

### Gaps
- No peeling paper found using Force Dimension or 3D Systems Touch haptic devices on Franka; no comparison of those devices vs SpaceMouse for peeling.
- No peeler-specific (rather than knife) demo-count scaling curve was found; ForceVLA/RDP give single operating points (~50-60).

## Q5. What evaluation metrics should a credible peeling paper report?

### Takeaway
There is **no standard peeling benchmark or rubric**; each paper invents its own. The union of published metrics is: binary motion success without damage, continuous peel length (e.g. > 10 cm), peeled fraction of cucumber length normalized by demos, average peel length per stroke, minimum strokes to clean, human holistic score (0-9) with a success threshold, perceived peel-thickness category, robustness under perturbation, contact-force stability, and generalisation to unseen instances/categories, typically over 10-20 trials per condition. Mass-based flesh waste and segmentation-based coverage % are not reported by any paper found and would be a genuine contribution.

### Cited Findings
- ForceMimic: two criteria, (1) "motion correct" = any peel removed "without damaging the zucchini", (2) "continuous peel longer than 10 cm"; 20 trials for main methods; HybridIL 100% / 85% vs Raw DP 80% / 55%; failure taxonomy (excess force damaging flesh, no contact, short peel, peel breakage); force curves reported - [ForceMimic, arXiv 2410.07554](https://arxiv.org/html/2410.07554)
- RDP: peeling score = "proportion of the peeled cucumber skin to the total length of the cucumber, normalized by the average score of the demonstration data"; three test-time variations (no perturbation with random 6D pose, perturbation before contact, perturbation after contact), 10 trials each; data quality measured by peel length, stable-force ratio and rolling standard deviation of force (window 10) - [RDP, arXiv 2503.02881](https://arxiv.org/html/2503.02881v3)
- ForceVLA: cucumber evaluated over 15 trials "each involving 15 peeling strokes"; success = "substantial cumulative peel coverage"; also average peel length per stroke (ForceVLA 14.12 cm vs pi0-base w/ F 13.17 cm vs w/o F 10.27 cm) and minimum strokes to clean (7 vs 10 vs 14) - [ForceVLA, arXiv 2505.22159](https://arxiv.org/html/2505.22159)
- Berkeley knife peeling: stroke defined along the principal axis; qualitative 0-9 holistic human score (thickness, length, continuity), success if > 3; quantitative peel thickness from wrist view in six discrete categories ("nominal" = most desired); 10 trials per reported number; zero-shot transfer to zucchini (50%), pear (90%), daikon (80%) - [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)
- MORPHeus: peeled/unpeeled classifier from vision + force + vibration; user-preferred coverage (fully, half, top-peeled); 12 food items chosen as extremes of shape, texture, size, skin-flesh colour contrast, skin thickness, deformability; comparison against a human peeling one-handed with the same board - [MORPHeus, arXiv 2404.06570](https://arxiv.org/html/2404.06570v2)
- Knife-peeling authors flag food waste as a practical limitation and suggest reusable "surrogate produce" - [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)

### Inferences
- A credible cucumber paper should report, per condition, N >= 10-20 trials with confidence intervals, and at minimum: (a) stroke success (contact made, no gouge, no safety stop), (b) continuous strip length and per-stroke peel length (cm), (c) coverage % of the cucumber surface after K strokes, (d) flesh waste = mass(before) - mass(after) minus peel mass, or peel mass per cm^2 as a thickness proxy, (e) strokes and time to full coverage, (f) mean and std of contact force during strokes, (g) number of reflex/safety stops, (h) variation in cucumber diameter/curvature (measured with calipers) and out-of-distribution produce (zucchini as in the knife paper).
- Coverage % can be measured with a top camera and colour/SAM2 segmentation of dark green skin vs pale flesh while rotating the cucumber on the fixture (SAM2 masking is already used in the knife paper); this needs a calibration step and is not published as a peeling metric.
- Reporting both the built-in estimate and a wrist F/T force trace for the same trials would directly answer Q2 in the paper.

### Gaps
- No published rubric for coverage % by segmentation, peel thickness by weighing, or peel-depth consistency was found.
- No cucumber-specific variability protocol (diameter bins, firmness) was found in the passages read.

## Q6. What safety and practicality issues matter (blade safety, Franka reflexes, hygiene, produce cost, cleaning)?

### Takeaway
Use Franka's collision thresholds plus controller-side force bounding (reference limiting / error clipping, torque-rate limits) as layered protection; treat the robot reflex as the last line, not the controller. FDM-printed parts that touch food are not automatically food-safe (layer lines trap bacteria; EU 10/2011 compliance must be verified for the finished part), so keep printed parts away from the cut surface or use smoothed/coated or stainless/food-grade contact parts. Food waste and produce cost per trial are real but unquantified in the literature.

### Cited Findings
- libfranka `setCollisionBehavior` configures contact and collision thresholds: values between lower and upper thresholds are reported as contacts in RobotState; above the upper threshold a collision "triggers reflex (robot stops), latches until recovery" - [libfranka Robot class reference](https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html); [Peter Corke libfranka notes](https://petercorke.com/robotics/franka-emika-control-interface-libfranka/)
- Example thresholds used by ViSP's Franka wrapper: lower torque thresholds {25, 25, 22, 20, 19, 17, 14} Nm and lower Cartesian force thresholds {30, 30, 30, 25, 25, 25} (example values, not Franka defaults) - [ViSP vpRobotFranka source](https://visp-doc.inria.fr/doxygen/visp-daily/vpRobotFranka_8cpp_source.html)
- HIL-SERL bounds impedance-controller force by clipping pose error |e| <= Delta, otherwise a far reference "can lead to a hard collision or damage" - [HIL-SERL, arXiv 2410.21845](https://arxiv.org/html/2410.21845v3)
- CRISP safety features: joint torque and rate limits, exponential filtering of targets, tracking error clipping - [CRISP, arXiv 2509.06819](https://arxiv.org/html/2509.06819)
- FR3 v5.8 Desk API can programmatically clear safety violations and activate FCI (useful for long unattended evaluation batches) - [FR3 brochure, Feb 2026](https://franka.de/hubfs/Brochure_Franka%20Research%203_February%202026_English.pdf)
- ForceMimic failure mode: excessive force "resulted in damage to the zucchini" and once "broke the bottom of the zucchini" - [ForceMimic, arXiv 2410.07554](https://arxiv.org/html/2410.07554)
- FDM food safety: "a 3D-printed PLA object is not automatically food-safe"; layer lines retain contamination, E. coli and S. aureus adhere; under EU Regulation 10/2011 migration limits must be verified for the finished product and "no EU authority has certified FDM-printed PLA objects as food-safe" (vendor blog, secondary) - [Creality blog](https://www.creality.com/blog/is-3d-printed-pla-food-safe); some filaments (rPLA, rPETG) are declared compositionally compliant with EU 10/2011 - [Filamentive food-safe guide](https://www.filamentive.com/food-safe-filament-guide-ensuring-3d-printing-food-safety/)
- FR3 arm protection rating IP40, control IP20 (secondary source) - [Anton Robots FR3 review](https://antonrobots.com/franka-research-3-review/)
- Food waste is acknowledged as a limitation of real-world food manipulation research - [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1)

### Inferences
- Layering for a blade task: (1) policy action bounds and workspace box, (2) clipped impedance error so max spring force is a few times the nominal ~10 N peel force, (3) torque-rate limits, (4) Franka contact/collision thresholds set above peel forces but well below defaults that allow large impacts, (5) external enable/stop device in the operator's hand, (6) cut-resistant gloves for anyone holding or loading produce (RDP had a human hand next to the blade).
- IP40 means no splashing: wipe cucumber juice off the arm, use a drip tray, keep water-cleaning to removable fixture parts.
- Treat produce as non-food after experiments (lab hygiene, not food-grade handling) unless the whole tool chain is food-grade; this sidesteps 10/2011 compliance for printed parts.

### Gaps
- Franka's official default collision thresholds and recommended values for tool-contact tasks were not retrieved.
- No source found for cucumber price per trial in Germany, nor for cleaning protocols used by peeling papers.
- No published blade-guard design for robot peelers was found.

## Q7. What does the extra hardware cost in EUR?

### Takeaway
Sourced prices: Bota SensONE Gen A **CHF 4,882.50**, Rokubi Gen A **CHF 5,407.50**, MiniONE Gen A **CHF 3,045.00** (Bota shop); Bota SensONE T15 FR3 kit **USD 8,198.38** (Vention); ATI Mini45 from **USD 6,675.00** (US distributor, more with options); RealSense D405 **EUR 327.95** (MyBotShop.de) to **EUR 414.96 incl. VAT** (Generation Robots); Meta Quest 3 ~**USD 500** (RDP paper). The wrist F/T sensor dominates the add-on budget; cameras, printed parts, peelers and produce are small by comparison but were not all priced in this session.

### Cited Findings
- Bota shop list prices: MiniONE Gen A (USB) CHF 3,045.00; Rokubi Gen A (EtherCAT, 3 m cable and accessories) CHF 5,407.50; SensONE Gen A (EtherCAT in/out, 3 m cable and accessories) CHF 4,882.50 - [Bota Systems shop](https://shop.botasys.com/shop/category/force-torque-sensors-4)
- Bota SensONE T15 Gen A kit for Franka FR3 with gripper extension cable: USD 8,198.38, ships in ~3 weeks - [Vention](https://vention.com/parts/bota-systems-sensone-t15-gen-a-6-axis-ft-sensor-kit-for-franka-fr3-with-gripper-extension-cable-4909)
- ATI Mini45 (9105-TW-MINI45-R, IP65 variant listing): USD 6,675.00, USD 9,325.00 with selected options - [e-motion supply](https://www.e-motionsupply.com/product_p/9105-tw-mini45-r-5-ip65.htm); ATI states contact sales for pricing - [ATI Mini45 page](https://www.ati-ia.com/products/ft/ft_models.aspx?id=mini45)
- RealSense D405: EUR 327.95 at MyBotShop.de - [MyBotShop.de](https://www.mybotshop.de/Intel-RealSense-Depth-Camera-D405); EUR 414.96 incl. VAT / EUR 345.80 excl. VAT at Generation Robots - [Generation Robots DE](https://www.generationrobots.com/de/403953-intel-realsense-depth-camera-d405.html); USD 272.00 on the RealSense store - [RealSense compare page](https://www.realsenseai.com/compare-all-cameras/); idealo listing up to EUR 550.00 - [idealo](https://www.idealo.de/preisvergleich/Liste/124245359/intel-realsense-d405.html)
- D405 works best at 7-50 cm range - [DigiKey DE](https://www.digikey.de/de/product-highlight/i/intel-realsense/d405-depth-camera)
- Meta Quest 3 for TactAR: "$500" - [RDP, arXiv 2503.02881](https://arxiv.org/html/2503.02881v3)

### Inferences
- Indicative add-on BOM (only sourced lines priced; CHF/USD not converted because no exchange rate was fetched):
  - Wrist F/T: Bota SensONE Gen A CHF 4,882.50 (or FR3 kit USD 8,198.38 via Vention; ATI Mini45 from USD 6,675). Buying direct from Bota (Zurich) is likely the cheapest EU-region route; confirm FR3 adapter plate is included.
  - Wrist cameras: 2x RealSense D405, about EUR 656-830 total at the German retailers above.
  - Teleop: SpaceMouse (not priced here) is the evidence-backed choice; Quest 3 (~USD 500) only if doing TactAR-style AR.
  - Third-person camera (D435/D435i), peeler(s), printed or machined adapter, between-centres fixture, kitchen scale and calipers, produce: not priced in this session, mark TBD.
- If no F/T is bought, total add-on cost is roughly the two cameras plus fixtures, i.e. under ~EUR 1,500 is plausible, but this total is not sourced.

### Gaps
- No EU price found for ATI Mini45, Robotiq FT 300-S, RealSense D435/D435i, 3Dconnexion SpaceMouse, peelers or printed parts.
- No FR3 robot price found.
- Currency conversion CHF/USD -> EUR not performed (no rate source fetched).
