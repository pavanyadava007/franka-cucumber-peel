# Force-aware and visuo-tactile policy learning for contact-rich tool manipulation (policy layer for Franka cucumber peeling)

Checked 2026-11-02. Sources are arXiv abstract and HTML pages, project pages, the GitHub API (licence, stars, issues), the Hugging Face API and proceedings pages. Venues are only called "verified" when a primary source states them. Lines marked "(inference)" are engineering judgement, not sourced claims. 2023 and older work is flagged as background. Five parallel verification passes did the reading. I re-checked the decision-critical 2026 claims (FAWAM, Dream-Tac, How to Peel) and the licences directly.

## Method-by-method comparison (the objective)

### Takeaway
Six learned force- or tactile-aware policies have published CUCUMBER peeling results with a handheld peeler: RDP, FoAR, ForceVLA, Dream-Tac, FAWAM and TouchGuide (ILA too, peeling tool not confirmed). ForceMimic peels zucchini. Only two of these ran on a Franka: FAWAM on FR3 with a wrist ATI sensor (19/20, no code) and Dream-Tac on a Panda with tactile fingertips (85%, Apache-2.0 code but a 2B video world model). The open-source, single-GPU-class methods with peeling evidence (RDP, ForceMimic, FoAR) were all run on Flexiv Rizon arms.

### Cited Findings

#### Master table (details and sources per method follow below)

| Method | Venue (status) | Robot (Franka?) | Force/tactile input | Action space and low-level handoff | Demos / collection | Peel / wipe / cut result | Compute | Code / licence |
|---|---|---|---|---|---|---|---|---|
| RDP | RSS 2025 (arXiv comment) | 2x Flexiv Rizon 4 (paper); FR3 path in repo, lightly validated | GelSight Mini, MCTac, or TCP wrench estimated from joint torques | Relative EE trajectory; slow latent diffusion 1-2 Hz, fast tokenizer decoder 24 Hz, interpolated to >500 Hz Flexiv RDK TCP control | 60 peel, 80 wipe; TactAR (Quest 3 AR force arrows) | Cucumber peel score 0.95 (force) vs DP 0.44 | RTX 4090; LDP 100 ms, AT <1 ms; user reports >20 GB VRAM to train | MIT, data + checkpoints on HF |
| ImplicitRDP | RA-L 2026 (arXiv comment) | Flexiv Rizon 4s | 6-axis wrist F/T | Relative Cartesian + virtual target + adaptive stiffness k in [200, 10000] N/m, impedance controller | 40 kinesthetic | No peel/wipe (box flip 18/20, switch 18/20) | RTX 4090, fast step <30 ms | MIT, data + model on HF |
| ForceMimic / HybridIL | ICRA 2025 (verified) | 2x Flexiv Rizon 4 (one learned, one rule-based holder) | Handheld ForceCapture (6-axis F/T, T265, L515) | Pose + wrench output; switches to hybrid force/position above 6 N (Flexiv RDK) | 15 zucchinis, 438 peel segments, robot-free handheld device | Zucchini peel >10 cm: 85% vs DP 55% | Not stated | MIT, data on Google Drive |
| FoAR | RA-L 2025 (repo BibTeX DOI only) | Flexiv Rizon | OptoForce F/T ~100 Hz | EE pose, plain position control, about 10 Hz chunks | 50 peel, 50 wipe, 40 chop; haptic teleop (RH20T style) | Cucumber peel score 0.756 vs DP 0.386; wipe 0.875 | 2x A100 train, RTX 3090 deploy | No licence |
| ForceVLA | NeurIPS 2025 (verified proceedings) | Flexiv Rizon 7-DoF | Estimated external TCP wrench | pi0 + force MoE; TCP pose + gripper; rate/controller not stated | 244 traj total, 43 cucumber peel; Quest 3 VR | Peel length 14.12 cm/stroke vs 10.27 (pi0 no F); 5-task avg 60.5% vs 37.3% | 8x RTX 4090 node; inference >8 GB, LoRA >22.5 GB | Apache-2.0, data on HF; training code reported broken |
| TA-VLA | CoRL 2025 (arXiv comment) | Cobot Magic ALOHA dual-arm (AgileX), ROKAE | Joint torque from motor current | pi0, joint positions, pure position control | 400 per task | No peel/wipe/cut | 4x L20 train; 4090 ~94 ms | Apache-2.0; models/full data not released |
| ACP | ICRA 2025 (README citation) | UR5e | ATI mini-45 wrist F/T | Reference pose + virtual target + 1 stiffness scalar; 500 Hz admittance | 200 wiping; kinesthetic with haptic feedback | Vase wiping 93.75% vs 43.75% compliant baseline | Not stated | MIT, data + checkpoints |
| UMI-FT | ICRA 2026 (project page) | UR5e | CoinFT per finger at 360 Hz | ACP-style 21-D (ref pose, virtual target, stiffness, width, grasp force); 500 Hz admittance | 275 wiping; handheld iPhone UMI | Whiteboard wiping 92% vs DP w/ F 28% | Not stated | MIT, hardware doc; no hosted data |
| FACTR | RSS 2025 (project BibTeX) | Franka Panda (1 or 2 arms) | Franka external joint torques | Absolute joint positions, chunk 100 | 50 per task; low-cost bilateral force-feedback leader arm | No peel; rolling dough 90/80 vs vision-only 0/0 | 1x RTX 4090, 2 to 6 h | Apache-2.0 (3 repos); Franka follower node not released |
| HIL-SERL | Science Robotics 2025 (verified) | Franka (Panda + FR3) | EE wrench in proprio (source not named) | 6D EE twist at 10 Hz into 1 kHz Cartesian impedance with reference limiting; feedforward wrench for dynamic tasks | 20-30 demos + SpaceMouse interventions + 1 to 6 h real RL | No peel/wipe; about 100% on 13 tasks | 1x RTX 4090 | Apache-2.0; ROS1 |
| Feel the Force | RSS 2025 workshop only | Franka Panda | AnySkin tactile, human glove | EE pose + gripper + desired grip force, 6 Hz | 30 human glove demos | No surface contact; 77% (58/75) grasp tasks | Not stated | No licence |
| DexForce | RA-L 2025 | Allegro hand (arm not stated) | CoinFT fingertip F/T | Force-shifted fingertip targets + Cartesian impedance | Kinesthetic | No surface contact; 76% avg | Not stated | No code |
| 3D-ViTac | CoRL 2024 (verified) | Bimanual ALOHA-style (arm not named) | FlexiTac piezoresistive | Joint states, diffusion | 30-50 | No peel/wipe | Not stated | Hardware CC BY-NC 4.0; training code not released |
| VT-Refine | CoRL 2025 (repo) | WidowX ALOHA 2, Kinova Gen3 | FlexiTac | Diffusion + sim RL fine-tune (DPPO) | 30 real + sim RL | Assembly only | Not stated | NVIDIA non-commercial |
| ViTaMIn / ViTaMIn-B | arXiv only | Rokae xMate ER3 Pro | Camera-based tactile fingers | Diffusion, 10 Hz | 87-345 handheld demos | Beaker wiping (B), numbers in figure only | Not stated | ViTaMIn: none; B: no licence |
| pi0 / pi0.5 | RSS 2025 / CoRL 2025 | Franka among many | None native | Action chunks H=50, up to 50 Hz; DROID Franka uses joint velocity at 15 Hz | 5 to 100+ h per task fine-tune | No peel results | Inference >8 GB, LoRA >22.5 GB, full >70 GB | Apache-2.0 |
| OpenVLA-OFT | RSS 2025 (arXiv comment) | ALOHA, LIBERO sim | None | Parallel-decoded chunks (K=8 or 25) | 20-300 | No contact-rich | Train min 25.6 GB; infer 15.9-18 GB | MIT |
| FAWAM | arXiv only (2606.08555) | **Franka FR3** | ATI Axia80-M8 wrist F/T, 120 Hz | GE-Act world action model, joint-velocity chunks; 10 Hz force-guided residual corrector; impedance control | 90 demos + 20 interventions; FACTR leader force-feedback teleop | **Peel Cucumber 19/20** vs TA-VLA 17, ForceVLA 14, pi0.5 8 | 8x A100 | No code (project page only) |
| Dream-Tac | arXiv only (2606.08737) | **Franka Emika Panda** | 2x Xense Photon tactile | Cosmos-Predict2-2B world action model, chunk H=20; controller not stated | 100 per task, SpaceMouse | **Peel Cucumber (peeler) 85%** | H200 training, ~5 Hz on A800 | Apache-2.0 |
| How to Peel with a Knife | arXiv only (2603.03280) | Kinova Gen3 | ATI mini45 | Delta EE pose 10 Hz into 500 Hz joint impedance | 50 cucumber, 150 apple, 200 potato; SpaceMouse | Knife peel 100% in-category; force ablation 80% vs 60% | Not reported | No licence |

#### Reactive Diffusion Policy (RDP)
- Title "Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation", arXiv 2503.02881. The arXiv comment says "Accepted to RSS 2025". Authors run Han Xue ... Cewu Lu. ([arXiv](https://arxiv.org/abs/2503.02881))
- RSS 2025 Best Student Paper Award finalist. ([GitHub README](https://github.com/xiaoxiaoxh/reactive_diffusion_policy))
- Affiliations: Tsinghua IIIS, Shanghai Qi Zhi Institute, Shanghai AI Lab / Shanghai Innovation Institute, and SJTU. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- Hardware: two Flexiv Rizon 4 arms with joint torque sensors, Flexiv Grav grippers, wrist RealSense D435, GelSight Mini (25 FPS, about $549) or a custom MCTac (about $50). ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
  - In the "Force" variant, the TCP wrench is estimated from Rizon joint torques via Flexiv RDK at 120 Hz, then downsampled to 24 FPS. No external F/T sensor is used. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- Action: relative end-effector trajectory, with no stiffness or wrench output. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
  - Slow policy: latent diffusion (LDP) at about 1-2 Hz.
  - Fast policy: asymmetric tokenizer (AT, 1D-CNN encoder + GRU decoder) that decodes actions at 24 FPS from tactile/force input.
  - Actions are interpolated and sent at more than 500 Hz through Flexiv RDK TCP control. The paper does not name impedance or admittance.
- The repo owner says the interpolator ran at 1000 Hz or more, and that other robots need an equivalent. ([issue #6](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/6))
- Data: 60 peeling, 80 wiping and 50 bimanual-lifting demos, collected with TactAR. TactAR is a Meta Quest 3 AR app that renders force as 3D arrows, with about 10 ms tracking and 10 ms rendering latency. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- Peeling task: the robot holds a peeler; the cucumber is held in midair by a human. Score = peeled length / cucumber length, normalised by the demo average. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- Peeling scores over all perturbations: RDP (Force) 0.95, RDP (GelSight) 0.90, RDP (MCTac) 0.88, DP 0.44, DP with tactile image 0.41. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- Wiping (eraser on a handheld vase): RDP (Force) 0.87 vs DP 0.57. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- Compute: inference on an RTX 4090 is DP 120 ms, LDP 100 ms, AT under 1 ms. Training time is not stated. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
  - A user (not a maintainer) reports that LDP training needs more than 20 GB of free VRAM. ([issue #10](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/10))
- Release: MIT, Copyright 2025 Han Xue, Wendi Chen, Yuan Fang, with third-party MIT / CreativeML Open RAIL-M / Apache-2.0 texts. ([LICENSE](https://raw.githubusercontent.com/xiaoxiaoxh/reactive_diffusion_policy/main/LICENSE))
  - Checkpoints and dataset are on Hugging Face. The repo has 395 stars and includes a Franka setup doc. ([GitHub](https://github.com/xiaoxiaoxh/reactive_diffusion_policy), [HF dataset](https://huggingface.co/datasets/WendiChen/reactive_diffusion_policy_dataset))
- No simulation. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))

#### ImplicitRDP (RDP follow-up)
- arXiv 2512.10946, "Accepted to RA-L 2026". Wendi Chen, Han Xue ... Cewu Lu (SJTU, SII, Noematrix). ([arXiv](https://arxiv.org/abs/2512.10946))
- Hardware and control: Flexiv Rizon 4s with a 6-axis wrist F/T sensor, under an impedance controller. ([arXiv HTML](https://arxiv.org/html/2512.10946v2))
- Action: relative Cartesian action + virtual target (x_vt = x_real + K^-1 f_ext) + an adaptive stiffness scalar k in [200, 10000] N/m. ([arXiv HTML](https://arxiv.org/html/2512.10946v2))
- Data: 40 kinesthetic demos per task. ([arXiv HTML](https://arxiv.org/html/2512.10946v2))
- Results out of 20 (ImplicitRDP / RDP / DP): box flipping 18 / 16 / 0, switch toggling 18 / 10 / 8. ([arXiv HTML](https://arxiv.org/html/2512.10946v2))
- No peeling or wiping. ([arXiv HTML](https://arxiv.org/html/2512.10946v2))
- Release: MIT (LICENSE file, Copyright 2026 Wendi Chen). Dataset and model are on HF. ([LICENSE](https://raw.githubusercontent.com/Chen-Wendi/ImplicitRDP/main/LICENSE), [project](https://implicit-rdp.github.io))

#### ForceMimic / HybridIL
- arXiv 2410.07554; the arXiv comment says "accepted by ICRA 2025". Wenhai Liu, Junbo Wang, Yiming Wang ... Cewu Lu, all SJTU. ([arXiv](https://arxiv.org/abs/2410.07554))
- ForceCapture (handheld, robot-free device). ([arXiv HTML](https://arxiv.org/html/2410.07554v3))
  - Sensing: 6-axis F/T between handle and tool at 1000 Hz, T265 SLAM at 200 Hz, external L515 RGB-D at 30 Hz.
  - Build: printed parts plus encoder cost about $50; weight 0.8 kg.
- The README says the F/T is "an inner product by Flexiv", replaceable by an ATI, and that the T265 and L515 are discontinued. ([forcecapture README](https://github.com/ForceMimic/forcecapture))
- HybridIL: a modified Diffusion Policy on point clouds that outputs pose AND wrench for 20 steps. Force is an output only, not an input. ([arXiv HTML](https://arxiv.org/html/2410.07554v3), [issue #1](https://github.com/ForceMimic/hybridil/issues/1))
- Execution: below 6 N predicted force, an IK position primitive. Above 6 N for consecutive steps, a hybrid force-position primitive in which force is projected orthogonal to the motion direction (Flexiv RDK). The policy and controller rates are not stated. ([arXiv HTML](https://arxiv.org/html/2410.07554v3))
- Setup: two Flexiv Rizon 4 arms. The left arm holds the zucchini under rule-based control; the right arm runs HybridIL with a fixed peeler. ([arXiv HTML](https://arxiv.org/html/2410.07554v3))
- Data: 15 zucchinis, giving 438 peeling segments and 30,199 action sequences. Collection takes about 5 min per zucchini with ForceCapture vs more than 13 min with force-feedback teleop. ([arXiv](https://arxiv.org/abs/2410.07554), [arXiv HTML](https://arxiv.org/html/2410.07554v3))
- Results, continuous peel longer than 10 cm: HybridIL 85% (17/20), Raw DP 55% (11/20), Force DP 10%, Force+Hybrid DP 20%. ([arXiv HTML](https://arxiv.org/html/2410.07554v3))
  - Motion-correct rates: HybridIL 100%, Raw DP 80%.
  - Mean contact force: HybridIL about 9 N vs Raw DP about 20 N.
  - The Force DP and Force+Hybrid DP rows had only 10 trials.
- Release: MIT. hybridil has 48 stars, forcecapture 37; both last pushed 2025-03-05. Onshape CAD is provided for the fixed-tool version. ([hybridil](https://github.com/ForceMimic/hybridil), [forcecapture](https://github.com/ForceMimic/forcecapture))
- No simulation. ([arXiv HTML](https://arxiv.org/html/2410.07554v3))

#### FoAR
- arXiv 2411.15753. The RA-L 2025 venue (vol 10 no 6, DOI 10.1109/LRA.2025.3560871) comes only from the repo BibTeX. The repo description adds IROS 2025. Zihao He, Hongjie Fang ... Hao-Shu Fang, Cewu Lu (SJTU). ([GitHub](https://github.com/Alan-Heoooh/FoAR), [project](https://tonyfang.net/FoAR/))
- Hardware: Flexiv Rizon with a Dahuan AG-95 gripper, a front RealSense D435, and an OptoForce F/T at about 100 Hz. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- Architecture: RISE sparse point-cloud policy, a 2 s force history encoder, and a future contact predictor whose probability gates the force features. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- Control: EE pose actions at about 10 Hz under plain position control. A reactive nudge of 0.006 m is applied when contact is predicted but force is below threshold. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- Data: 50 demos each for peeling and wiping (40 for chopping), via RH20T-style haptic teleop. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- Cucumber peeling score (peeler grasped by the robot), 20 trials: FoAR 0.756, DP 0.386, RISE 0.377, ACT 0.120. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- Other tasks:
  - Whiteboard wiping: FoAR 0.875 vs RISE 0.500, DP 0.400.
  - Chopping: 3.9 vs 1.8 segments.
  ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- Compute: trained on 2x A100, deployed on an RTX 3090. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- Release: code is public, but the GitHub API returns license null (no LICENSE file). Sample peel and wipe data are on Google Drive. ([GitHub](https://github.com/Alan-Heoooh/FoAR))

#### ForceVLA
- arXiv 2505.22159, NeurIPS 2025 main track (verified in the proceedings). Jiawen Yu ... Cewu Lu (SJTU and others). ([NeurIPS proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/8633b46e12cc5f2ee1f05a6ca2c65b38-Abstract-Conference.html))
- Hardware: Flexiv Rizon 7-DoF. Force is an "estimated external wrench" on the TCP; no sensor model is named. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
- Model: pi0 base. The 6-D wrench becomes one token, fused after the VLM by a 4-expert top-1 MoE. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
- Action: TCP pose + gripper. The policy rate and low-level controller are not stated. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
- Data: Quest 3 VR teleop, 244 trajectories over 5 tasks. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
  - The HF folders show 43 cucumber-peeling trajectories. ([HF API](https://huggingface.co/api/datasets/qiaojunyu/ForceVLA-real-data))
- Cucumber peeling: no success rate is reported, only peel length per stroke and strokes to clean. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
  - ForceVLA: 14.12 cm, 7 strokes.
  - pi0-base with F: 13.17 cm, 10 strokes.
  - pi0-base without F: 10.27 cm, 14 strokes.
- Five-task average success: ForceVLA 60.5% vs 37.3% for pi0-base without F. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
- Compute: training on an 8x RTX 4090 node; single task 10k steps in about 9 h on 1 GPU. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
  - Memory: inference over 8 GB, LoRA over 22.5 GB, full fine-tune over 70 GB. ([GitHub](https://github.com/ft-robotic/ForceVLA))
- Release: Apache-2.0 code, a JAX openpi fork, 116 stars. The dataset is Apache-2.0 in LeRobot parquet format. No official checkpoints were found. ([GitHub](https://github.com/ft-robotic/ForceVLA), [HF dataset](https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data))

#### TA-VLA
- arXiv 2509.07962; the arXiv comment says "Accepted to CoRL 2025". Zongzheng Zhang ... Hao Zhao (BAAI, Tsinghua AIR, NTU). ([arXiv](https://arxiv.org/abs/2509.07962))
- Hardware: Cobot Magic ALOHA dual-arm robot; joint torque derived from motor current; no F/T sensor. ([arXiv HTML](https://arxiv.org/html/2509.07962v1))
- Model and action: pi0 with a torque token in the decoder, plus an auxiliary torque prediction. Actions are joint positions. ([arXiv HTML](https://arxiv.org/html/2509.07962v1))
  - Deployment is pure position control. ([issue #10](https://github.com/ZZongzheng0918/TA-VLA/issues/10))
- Data: 400 demos per task. ([arXiv HTML](https://arxiv.org/html/2509.07962v1))
- Tasks: no peeling, wiping or cutting. Charger plugging: pi0 0/20 vs TA-VLA 17/20. ([arXiv HTML](https://arxiv.org/html/2509.07962v1))
- Compute: training on 4x L20 with LoRA; about 94 ms inference on an RTX 4090. ([arXiv HTML](https://arxiv.org/html/2509.07962v1))
- Release: Apache-2.0 code; models and the full dataset are not released. ([GitHub](https://github.com/ZZongzheng0918/TA-VLA))

#### Adaptive Compliance Policy (ACP) and UMI-FT
- ACP: arXiv 2410.09309. ICRA 2025, pp 4829-4836, per the README citation. Yifan Hou ... Shuran Song (Stanford, TRI). ([GitHub](https://github.com/yifan-hou/adaptive_compliance_policy), [project](https://adaptive-compliance.github.io/))
- ACP hardware: UR5e, ATI mini-45 F/T (up to 7000 Hz), GoPro fisheye. ([arXiv HTML](https://arxiv.org/html/2410.09309v2))
- ACP action and control: per arm, a 9D reference pose + 9D virtual target pose + 1 stiffness scalar in the compliant direction. Execution uses admittance control with 500 Hz pose commands. ([arXiv HTML](https://arxiv.org/html/2410.09309v2))
- ACP data: kinesthetic teaching with haptic feedback; 200 wiping demos and 230 flipping episodes. ([arXiv HTML](https://arxiv.org/html/2410.09309v2))
- ACP results: ([arXiv HTML](https://arxiv.org/html/2410.09309v2))
  - Vase wiping: ACP 93.75% vs compliant baseline 43.75%. The stiff baseline was dropped because it broke the tool.
  - Item flipping: 96% vs 23% / 14%.
- ACP release: MIT, with checkpoints and data zips; 144 stars. ([GitHub](https://github.com/yifan-hou/adaptive_compliance_policy))
- UMI-FT: arXiv 2601.09988, ICRA 2026 per the project page. Hojung Choi, Yifan Hou ... Shuran Song. ([arXiv HTML](https://arxiv.org/html/2601.09988v1), [project](https://umi-ft.github.io/))
  - Hardware: CoinFT 6-axis sensors per finger at 360 Hz, an iPhone 15 Pro for pose and RGB-D, and a UR5e. ([arXiv HTML](https://arxiv.org/html/2601.09988v1))
  - Action and control: ACP-style action plus grasp force; 500 Hz admittance. ([arXiv HTML](https://arxiv.org/html/2601.09988v1))
  - Whiteboard wiping: 92% (23/25) vs DP with force 28%, DP 16%. ([arXiv HTML](https://arxiv.org/html/2601.09988v1))
  - Release: MIT. The README supports UR robots only. ([GitHub](https://github.com/real-stanford/UMI-FT))
  - Caveat from the paper: compliance gave little benefit on zucchini skewering because the zucchini itself is compliant. ([arXiv HTML](https://arxiv.org/html/2601.09988v1))

#### FACTR
- arXiv 2502.17432; RSS 2025 per the project BibTeX (DOI 10.15607/RSS.2025.XXI.079). Jason Jingzhou Liu, Yulong Li ... Deepak Pathak (CMU). ([project](https://jasonjzliu.com/factr/))
- Hardware: Franka Panda follower(s) with an OpenManipulator-X gripper. ([arXiv HTML](https://arxiv.org/html/2502.17432v2))
  - Leader: a low-cost bilateral arm from Dynamixel servos, about $1,229.95, with force feedback and gravity compensation.
  - Force input: Franka external joint torques; no F/T sensor.
- Action: absolute joint positions, ACT with chunk 100. Training follows a curriculum that blurs the visual features so the policy attends to force. ([arXiv HTML](https://arxiv.org/html/2502.17432v2))
- Data: 50 demos per task. ([arXiv HTML](https://arxiv.org/html/2502.17432v2))
- Results, average unseen-object success: FACTR 87.5 vs vision+force 61.2 vs vision-only 21.3. Rolling dough: 90/80 vs 0/0 for vision-only. ([arXiv HTML](https://arxiv.org/html/2502.17432v2))
- Compute: one RTX 4090, 2 to 6 h. ([arXiv HTML](https://arxiv.org/html/2502.17432v2))
- Release: Apache-2.0 for training, teleop and hardware repos. ([factr](https://github.com/RaindragonD/factr), [factr_teleop](https://github.com/RaindragonD/factr_teleop), [hardware](https://github.com/JasonJZLiu/FACTR_Hardware))
- Leader loop configured at 500 Hz. ([teleop config](https://github.com/RaindragonD/factr_teleop))

#### HIL-SERL and SERL
- HIL-SERL: Science Robotics vol 10 issue 105, eads5033 (2025), DOI 10.1126/scirobotics.ads5033. Jianlan Luo, Charles Xu, Jeffrey Wu, Sergey Levine (UC Berkeley). ([Science Robotics](https://www.science.org/doi/10.1126/scirobotics.ads5033))
- Control: 10 Hz policy over 6D EE twist into a 1000 Hz Cartesian impedance controller with reference limiting. Jenga whipping and object flipping use a feedforward wrench action. ([arXiv HTML](https://arxiv.org/html/2410.21845v3))
- Training: RLPD with 20 to 30 demos plus SpaceMouse interventions, 1 to 6 h per task. Success rises from 49.7% (BC/HG-DAgger) to 100% average. ([arXiv HTML](https://arxiv.org/html/2410.21845v3))
- Compute: a single RTX 4090. ([arXiv HTML](https://arxiv.org/html/2410.21845v3))
- The bimanual setup used "a Panda and a FR3". ([robot infra README](https://github.com/rail-berkeley/hil-serl/blob/main/serl_robot_infra/README.md))
- HIL-SERL code: Apache-2.0, 1568 stars. ([GitHub](https://github.com/rail-berkeley/hil-serl))
- serl_franka_controllers: MIT, ROS Noetic, a Cartesian impedance controller with reference limiting in the real-time loop. ([GitHub](https://github.com/rail-berkeley/serl_franka_controllers))
- SERL: ICRA 2024 per the arXiv comment; Franka; 10 Hz policy, 1 kHz impedance. ([arXiv](https://arxiv.org/abs/2401.16013))

#### Feel the Force, DexForce (grasp/hand force, background for peeling)
- Feel the Force: arXiv 2506.01944. Shown at the RSS 2025 RoboReps workshop; no CoRL main-track listing was found. Ademi Adeniji ... Lerrel Pinto (NYU). ([project](https://feel-the-force-ftf.github.io/), [author page](https://ademiadeniji.github.io/publications))
  - Hardware: Franka Panda with an AnySkin tactile sensor.
  - Action: EE pose + gripper + desired grip force at 6 Hz. Force only regulates gripper squeeze.
  - Results: 77% from 30 human glove demos.
  ([arXiv HTML](https://arxiv.org/html/2506.01944v1))
  - No licence file. ([GitHub](https://github.com/feel-the-force-ftf/feel-the-force))
- DexForce: arXiv 2501.10356, RA-L 2025 (vol 10 no 6, DOI 10.1109/LRA.2025.3568318). Claire Chen ... Jeannette Bohg (Stanford). ([project](https://clairelc.github.io/dexforce.github.io/))
  - Method: force-informed target x_f = x_o + k_f f, executed under Cartesian impedance with k_p = 310.
  - Hardware and results: Allegro hand, 76% average.
  ([arXiv HTML](https://arxiv.org/html/2501.10356v2))
  - No code. ([project](https://clairelc.github.io/dexforce.github.io/))

#### 3D-ViTac, VT-Refine, ViTaMIn
- 3D-ViTac: arXiv 2410.24091, CoRL 2024 (OpenReview). Binghao Huang ... Yunzhu Li (Columbia, UIUC, UW). ([OpenReview](https://openreview.net/forum?id=bk28WlkqZn))
  - Hardware: FlexiTac piezoresistive pads, 16x16 taxels, about $20 per pad and board.
  - Policy: joint-state diffusion policy with a PointNet++ backbone. No peeling or wiping.
  ([arXiv HTML](https://arxiv.org/html/2410.24091))
  - Release: the hardware repo is CC BY-NC 4.0; the training code repo is a readme only. ([hardware](https://github.com/binghao-huang/3D-ViTac_Tactile_Hardware), [training](https://github.com/binghao-huang/3d-vitac-training-code))
- VT-Refine: arXiv 2510.14930, CoRL 2025 per the repo. ([GitHub](https://github.com/NVlabs/vt-refine))
  - Method: 30 real demos, then DPPO RL fine-tuning in Isaac Gym + TacSL.
  - Results: visuo-tactile real success after fine-tuning is 0.75 to 0.95 per asset.
  - The authors state: "Isaac Gym and the tactile simulation do not support deformable objects".
  ([arXiv HTML](https://arxiv.org/html/2510.14930v1))
  - Licence: NVIDIA research-only, non-commercial. ([LICENSE](https://raw.githubusercontent.com/NVlabs/vt-refine/main/LICENSE))
- ViTaMIn: arXiv 2504.06156, arXiv only, Tsinghua and UC Berkeley (Rui Chen, Jing Xu), not the Cewu Lu lab. Rokae arm; code "coming soon". ([arXiv](https://arxiv.org/abs/2504.06156), [project](https://chuanyune.github.io/ViTaMIn_page/))
- ViTaMIn-B: arXiv 2511.05858. Uses Quest 3 controller tracking (valid-demo rate 1.0 vs 0.16 for SLAM) and includes a beaker-wiping task. ([arXiv HTML](https://arxiv.org/html/2511.05858))
  - Code has no licence. ([GitHub](https://github.com/chuanyune/ViTaMIn-B_code))

#### Diffusion Policy, UMI (baselines)
- Diffusion Policy: arXiv 2303.04137, RSS 2023 and IJRR 2024. The real-world sauce-spreading task with a spoon is continuous tool-surface contact, but uses no force. UR5. ([arXiv](https://arxiv.org/abs/2303.04137), [project](https://diffusion-policy.cs.columbia.edu/)) Background (2023).
- UMI: arXiv 2402.10329, RSS 2024. ([project](https://umi-gripper.github.io/))
  - Robots: UR5/UR5e and Franka FR2; the Franka cross-embodiment cup task scored 18/20.
  - No F/T. Dish washing with sponge wiping: 70%.
  ([arXiv HTML](https://arxiv.org/html/2402.10329))
- UMI ships a Franka Cartesian impedance path through Polymetis: Kx = [750,750,750,15,15,15], Kxd = [37,37,37,2,2,2], at 1000 Hz. ([franka_interpolation_controller.py](https://raw.githubusercontent.com/real-stanford/universal_manipulation_interface/main/umi/real_world/franka_interpolation_controller.py))

#### VLA fine-tuning baselines
- pi0: arXiv 2410.24164, "Published in RSS 2025". 3.3B parameters, chunk H=50. Franka runs at 20 Hz. No force input. 73 ms inference on an RTX 4090. ([arXiv API](https://export.arxiv.org/api/query?id_list=2410.24164), [arXiv HTML](https://arxiv.org/html/2410.24164))
- pi0.5: CoRL 2025, PMLR 305:17-40. ([PMLR](https://proceedings.mlr.press/v305/black25a.html))
- openpi memory needs: inference over 8 GB, LoRA over 22.5 GB, full fine-tune over 70 GB. The PyTorch port does not support LoRA. Apache-2.0. ([openpi](https://github.com/Physical-Intelligence/openpi))
- openpi DROID Franka example: joint-velocity actions at 15 Hz. ([droid main.py](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py))
- OpenVLA-OFT: arXiv 2502.19645, "Accepted to RSS 2025". MIT. ([arXiv API](https://export.arxiv.org/api/query?id_list=2502.19645), [GitHub](https://github.com/moojink/openvla-oft))
  - Memory: training minimum 25.6 GB per GPU; inference 15.9 to 18.0 GB. ([project](https://openvla-oft.github.io/))
- Tactile-VLA (arXiv 2507.09160) fine-tunes pi0 with tactile input and a hybrid position-force controller. Whiteboard wiping: 80% vs pi0-base 40%. ([arXiv HTML](https://arxiv.org/html/2507.09160v1))

### Inferences
- (inference) On published peeling evidence plus reproducibility, RDP is the strongest single basis. It has a cucumber + peeler task, MIT code, released data and checkpoints, and an FR3 path in the repo. Its best variant uses a TCP wrench estimated from joint torques, which is the same kind of signal the Franka gives (O_F_ext_hat_K).
- (inference) Its weak point for a Franka is the action interface: pure relative pose, with compliance left to an unspecified 500 to 1000 Hz controller.
- (inference) ACP / UMI-FT / ImplicitRDP's "virtual target + stiffness" output is the best conceptual match to a Franka Cartesian impedance controller. A hybrid design (RDP-style slow/fast policy with an ACP/ImplicitRDP-style virtual-target + stiffness head) is a defensible thesis contribution, because no paper has published that combination on a peeler task.

### Gaps
- No source states control rates for ForceMimic, ForceVLA or TA-VLA, and none reports total collection hours for any peeling dataset.
- FAWAM per-task demo hardware details beyond FR3 + ATI + FACTR leader (for example the peeler mount and cucumber fixture) were not found.
- Dream-Tac's action parameterisation and low-level controller are not stated in the text read.

## Which method has the strongest published evidence on peeling or a peeling-like task?

### Takeaway
On a Franka, FAWAM has the strongest peeling number: 19/20 Peel Cucumber on an FR3 with a wrist ATI F/T, beating ForceVLA and TA-VLA reimplementations. But it is arXiv-only, has no code, and needs 8x A100. Among reproducible open-source methods, RDP (0.95 peel score, MIT) and ForceMimic/HybridIL (85% zucchini peels over 10 cm, MIT) are the strongest, both on Flexiv. The most rigorous peeling study overall (knife, Kinova) is "How to Peel with a Knife".

### Cited Findings
- FAWAM (arXiv 2606.08555v3, 7 Jun 2026, no venue on the abs page): Haotian He (Peking Univ.), Zeyu Yan, Qipeng Liu, Ning Guo, Wenzhao Lian (SJTU). ([arXiv](https://arxiv.org/abs/2606.08555))
  - Hardware: "All real-world experiments are conducted on a Franka Research 3 (FR3)", with "An ATI Axia80-M8 6-axis force/torque sensor ... mounted at the wrist", recording at 120 Hz with a 10 Hz Butterworth filter. ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
  - Data: collected with "a FACTR leader robot for force-feedback teleoperation"; the follower stays under impedance control. ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
  - Policy: a 10 Hz force-guided residual corrector on top of a GE-Act world action model. ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
  - Training: 90 demos per task plus 20 intervention episodes. Peeling varies the table height. ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
  - Peel Cucumber, 20 trials each: pi0.5 8/20, GE-Act 10/20, ForceVLA 14/20, TA-VLA 17/20, FEAM (no residual) 15/20, FAWAM 19/20. Four-task averages: 47.5%, 48.75%, 63.75%, 63.75%, 73.75% and 85.0%. ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
  - Compute: 8 NVIDIA A100. Only a project page is given, no code. ([arXiv HTML](https://arxiv.org/html/2606.08555v3), [project](https://fawam.github.io))
- Dream-Tac (arXiv 2606.08737, 7 Jun 2026, comment "16 pages, 13 figures", no venue): Peking Univ. (Shanghang Zhang). ([arXiv](https://arxiv.org/abs/2606.08737))
  - Hardware: "We evaluate Dream-Tac on a Franka Emika Panda robot", with two Xense Photon fingertip tactile sensors and two D435i cameras. ([arXiv HTML](https://arxiv.org/html/2606.08737))
  - Task: "Peel Cucumber: the robot uses a peeler to peel a cucumber; success requires producing a peeled strip longer than 5 cm." ([arXiv HTML](https://arxiv.org/html/2606.08737))
  - Results: Dream-Tac reaches 85% on Peel Cucumber, "slightly below ForceVLA". The six-task average is 83.3% vs Cosmos-Policy 51.7%, ForceVLA 50.8%, pi0.5 45.0% and pi0 30.8%. ([arXiv HTML](https://arxiv.org/html/2606.08737))
  - Compute: inference runs at about 5 Hz on an A800 with 10 denoising steps; caching reduces latency from 1109 ms to 619 ms. ([arXiv HTML](https://arxiv.org/html/2606.08737))
  - Code: Apache-2.0, 49 stars. ([GitHub](https://github.com/LYFCLOUDFAN/Dream-Tac))
- RDP peeling score: RDP (Force) 0.95, GelSight 0.90, MCTac 0.88, DP 0.44. The peel happens on a cucumber held in midair by a human, with perturbations before and after contact. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- ForceMimic/HybridIL, zucchini peel over 10 cm: 85% vs DP 55%. This is the only peeling paper whose policy outputs a target wrench that a hybrid force/position controller executes. ([arXiv HTML](https://arxiv.org/html/2410.07554v3))
- FoAR cucumber peeling score: 0.756 vs DP 0.386. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- ForceVLA cucumber peeling: 14.12 cm per stroke and 7 strokes to clean, vs pi0 without force 10.27 cm and 14 strokes. No success rate is given. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
- How to Peel with a Knife (arXiv 2603.03280, 3 Mar 2026, arXiv-only; Toru Lin ... Pieter Abbeel, Jitendra Malik, UC Berkeley). ([arXiv](https://arxiv.org/abs/2603.03280))
  - Abstract: "Using only 50-200 peeling trajectories, our system achieves over 90% average success rates on challenging produce including cucumbers, apples, and potatoes, with performance improving by up to 40% through preference-based finetuning." ([arXiv](https://arxiv.org/abs/2603.03280))
  - Hardware: Kinova Gen3, ATI mini45, two D405 cameras, SpaceMouse teleop. ([README](https://github.com/ToruOwO/how-to-peel))
  - Release: the reward-model and fine-tuning code is "coming soon". ([README](https://github.com/ToruOwO/how-to-peel))
  - Licence: the GitHub API reports none.
  - Results: in-category 100% for cucumber, apple and potato. Zero-shot cucumber policy on zucchini 50%. Potato modality ablation: RGB+D+F 80% vs no force 60%. ([arXiv HTML](https://arxiv.org/html/2603.03280))
- TouchGuide (arXiv 2601.20239; SJTU / Xense). ([arXiv HTML](https://arxiv.org/html/2601.20239))
  - Setup: bimanual cucumber peeling with a peeler on ARX5 arms and TacUMI data.
  - Mean scores: DP 0.500, RDP 0.740, pi0.5 0.785, TouchGuide on pi0.5 0.975. The table was garbled in extraction, so the grouping of these scores is uncertain.
- ILA (arXiv 2610.18855; NUS and others, dual ARX X5, Xense tactile). ([arXiv HTML](https://arxiv.org/html/2610.18855))
  - Cucumber peeling SR: FM-ILA + RL 85%, FM-tac 50%, DreamTac reimplementation 20%.
  - Tactile ablation: 50% with vs 20% without.
- FORCA (arXiv 2610.31841, UIUC, UR5e). A training-free LLM coding agent with F/T feedback, not a learned policy. Peeling ratio 81.1% vs vision-only 37.4%; contact loss 2.2% vs 40.6%. ([arXiv HTML](https://arxiv.org/html/2610.31841))
- ForceFlow (arXiv 2605.11048, xArm6). Peeling is qualitative only. Whiteboard cleaning 100% SR with force MAE 4.59 N vs ForceVLA 20.16 N. ([arXiv HTML](https://arxiv.org/html/2605.11048))
- Background, non-learned or older:
  - MORPHeus (arXiv 2404.06570): compliant assisted peeling of 12 food items. ([arXiv](https://arxiv.org/abs/2404.06570))
  - MIT "Vegetable Peeling: A Case Study in Constrained Dexterous Manipulation" (arXiv 2407.07884): in-hand reorientation to support peeling. ([arXiv](https://arxiv.org/abs/2407.07884))

### Inferences
- (inference) FAWAM is the strongest proof that wrist-F/T force-aware policies help peeling on exactly the target robot (FR3). Its baseline ordering (TA-VLA 17 and ForceVLA 14 vs pi0.5 8, out of 20) is useful evidence for a committee that force input roughly doubles peeling success over a vision-only VLA. It is a single-lab, arXiv-only result.
- (inference) The peeling metrics are not comparable across papers: strip over 5 cm (Dream-Tac), peel over 10 cm (ForceMimic), normalised length score (RDP, FoAR, TouchGuide), peel length per stroke (ForceVLA). A cucumber thesis should report several of these, ideally peel-length ratio and success at a strip-length threshold.
- (inference) RDP and FoAR peel a cucumber held by a human or grasped in air. ForceMimic uses a second arm under rule-based control, and FAWAM varies the table height. A fixtured cucumber with geometry variation is closest to FAWAM's and Dream-Tac's setups.

### Gaps
- FAWAM's cucumber fixture, peeler mount and per-task success criterion were not extracted from the text read.
- ForceVLA's per-task peeling success exists only in a figure.
- No independent reproduction of any peeling result was found.

## What action/control interface does each use, and which fits a Franka Cartesian impedance controller best?

### Takeaway
Interfaces fall into four groups:
1. Pure pose or joint actions over a stiff or position controller: RDP, FoAR, ForceVLA, TA-VLA, FACTR, pi0, OpenVLA-OFT.
2. Pose + virtual target + stiffness: ACP, UMI-FT, ImplicitRDP.
3. Pose + target wrench under hybrid force/position control: ForceMimic/HybridIL.
4. EE twist into a 1 kHz Cartesian impedance controller, plus feedforward wrench for dynamic tasks: HIL-SERL/SERL.

The virtual-target + stiffness family maps most directly onto the Franka Cartesian impedance law. HIL-SERL's serl_franka_controllers and UMI's Polymetis path are the existing Franka-side implementations.

### Cited Findings
- ACP: per-arm output of reference pose, virtual target pose and a stiffness scalar. Stiffness K = S K0 S^-1, with the compliant direction taken from the reference-to-virtual-target vector. Execution is admittance control on a UR5e at 500 Hz. ([arXiv HTML](https://arxiv.org/html/2410.09309v2))
  - Author advice: k_min should be 200-500 N/m and k_max about 5000 N/m. Labels come from an offline window over future force, so the policy softens before contact. ([issue #2](https://github.com/yifan-hou/adaptive_compliance_policy/issues/2))
- ImplicitRDP: virtual target x_vt = x_real + K^-1 f_ext plus an adaptive stiffness k in [200, 10000] N/m, executed by an impedance controller with a Cartesian PI term. ([arXiv HTML](https://arxiv.org/html/2512.10946v2))
- UMI-FT: ACP-style action with grasp force, 6D admittance at 500 Hz. ([arXiv HTML](https://arxiv.org/html/2601.09988v1))
- ForceMimic/HybridIL: pose + wrench prediction. Above a 6 N threshold it switches to a hybrid force-position primitive, with force projected orthogonal to motion (Flexiv RDK). ([arXiv HTML](https://arxiv.org/html/2410.07554v3))
- DexForce: shifts the commanded target by k_f times the measured force and executes it under Cartesian impedance (k_p = 310). Same virtual-target idea with fixed stiffness. ([arXiv HTML](https://arxiv.org/html/2501.10356v2))
- RDP: relative EE pose trajectories at 24 Hz, interpolated to more than 500 Hz on Flexiv TCP control; the controller type is not stated. ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
- FoAR: EE pose under plain position control. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- ForceVLA: TCP pose + gripper. ([arXiv HTML](https://arxiv.org/html/2505.22159v3))
- TA-VLA: joint positions, pure position control. ([issue #10](https://github.com/ZZongzheng0918/TA-VLA/issues/10))
- FACTR: absolute joint positions; the follower controller is not stated. The maintainers say they wrote their own Franka node and franka_ros "could be directly used". ([arXiv HTML](https://arxiv.org/html/2502.17432v2), [issue #1](https://github.com/RaindragonD/factr/issues/1))
- FAWAM: joint-velocity action chunks with a 10 Hz force-guided residual corrector. The follower stays "under impedance control for safe and compliant contact". ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
- HIL-SERL: 6D EE twist at 10 Hz into a 1000 Hz Cartesian impedance controller with reference limiting. Jenga whipping and flipping use feedforward wrench actions (Fx, Fz, tau). ([arXiv HTML](https://arxiv.org/html/2410.21845v3))
  - serl_franka_controllers: Cartesian impedance with reference limiting in the real-time loop; gains are tunable via rqt_reconfigure. It needs ROS Noetic and libfranka/franka_ros 0.8.0 or later. ([GitHub](https://github.com/rail-berkeley/serl_franka_controllers))
- UMI's Franka path: Polymetis start_cartesian_impedance with Kx = 750 N/m translational and update_desired_ee_pose at 1000 Hz. ([controller code](https://raw.githubusercontent.com/real-stanford/universal_manipulation_interface/main/umi/real_world/franka_interpolation_controller.py))
- RDP's Franka server originally put 7-D joint torques into the "TCP wrench" field. The maintainer confirmed the bug, and PR #32 converts it to a 6-D wrench via the Jacobian. ([issue #28](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/28))
- LeRobot has no core Franka robot. A third-party "lerobot-robot-franka-research3" plugin drives FR3 through franky with joint / Cartesian / impedance control. ([third-party robots doc](https://github.com/huggingface/lerobot/blob/main/docs/source/third_party_robots.mdx))

### Inferences
- (inference) Best fit for a Franka Cartesian impedance controller is "pose + virtual target + stiffness" (ACP / ImplicitRDP / UMI-FT). The torque law F = K(x_vt - x) - D xdot is exactly what a libfranka or franka_ros 1 kHz Cartesian impedance controller computes.
  - Required change: a controller that accepts a time-varying, possibly non-diagonal 3x3 translational K. The damping must be recomputed for each K, and setpoints interpolated between 10 to 24 Hz policy updates.
  - serl_franka_controllers (reference limiting, 1 kHz) is a good starting point.
- (inference) ACP's UR-derived k_max of about 5000 N/m is far above UMI's 750 N/m Franka gains. Stiffness labels should be regenerated with Franka-feasible bounds; the 200 to 2000 N/m range quoted by the ACP author for low-cost arms is a starting guess, not a verified Franka value.
- (inference) For peeling, the "virtual target below the surface by K^-1 f_desired" formulation gives an explicit, inspectable normal-force setpoint. It is also safer on a Franka than HybridIL-style explicit force control, because the Franka has no wrist F/T by default and explicit force control on estimated wrench is noisy.
  - HybridIL's wrench output can be converted to this form: x_vt = x + K^-1 f_pred.
- (inference) Pose-only methods (RDP, FoAR, ForceVLA) still work through a moderately compliant fixed-stiffness Cartesian impedance controller, as UMI shows. That is the lowest-effort path, but force regulation then depends implicitly on the fixed K.

### Gaps
- No paper reports the Franka-specific stiffness range used for a peeling task.
- Dream-Tac's controller and FACTR's follower controller are not stated.
- Polymetis maintenance status and FR3 support were not checked.

## Which have complete open-source code + hardware guide for a single master's student in 2 to 3 months, and what are the known reproduction pain points?

### Takeaway
Most complete releases: RDP (MIT; code, data, checkpoints, Franka doc, TactAR app), ACP (MIT; code, data, checkpoints, no hardware guide), UMI (MIT; full hardware guide, Franka path), HIL-SERL/SERL (Apache-2.0/MIT, Franka-native infra) and FACTR (Apache-2.0; Franka teleop hardware, but no follower node). ForceMimic/HybridIL is MIT but has discontinued sensors and Flexiv-specific control. FoAR, Feel the Force and How to Peel have no licence. ForceVLA's official training code is reported broken. FAWAM, DexForce and ViTaMIn have no usable code.

### Cited Findings

#### RDP
- 39 issues total, 1 open, when checked by the subagent; the API reports 2 open issues and PRs. ([issues](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues?q=is%3Aissue))
- Franka setup problems ([issue #31](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/31)):
  - ROS2 Humble (Python 3.10) and Polymetis (Python 3.8) conflict.
  - The new deoxys server is "not yet fully validated on a physical robot".
  - Single-arm calibration needed hand edits.
  - An unanswered port mismatch (8092 vs 8082).
- Wrench bug: the TCP wrench field held joint torques; fixed via PR #32. ([issue #28](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/28))
- GelSight Mini reached only about 18 fps on Ubuntu because of a USB 2.0 / MJPG cap. ([issue #13](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/13))
- Jitter on non-Flexiv arms; a high-rate interpolator is needed. ([issue #6](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/6))
- Franka doc: a separate real-time-kernel PC (NUC recommended) runs Polymetis, and FrankaServer should run as close to 1000 Hz as possible. ([franka_setup_instructions.md](https://raw.githubusercontent.com/xiaoxiaoxh/reactive_diffusion_policy/main/docs/franka_setup_instructions.md))
- Stack: ROS2 Humble, Ubuntu 22.04, PyTorch 1.13.1 + CUDA 11.7. ([GitHub](https://github.com/xiaoxiaoxh/reactive_diffusion_policy))

#### ACP and UMI
- ACP has 3 open issues. #4 (Aug 2026) suggests the force-encoder parameters may be missing from the optimizer; there is no maintainer reply. ([issues](https://github.com/yifan-hou/adaptive_compliance_policy/issues?q=is%3Aissue))
- ACP's C++ admittance stack (force_control, hardware_interfaces) is UR-specific. ([GitHub](https://github.com/yifan-hou/adaptive_compliance_policy))
- UMI has about 30 SLAM-related issues, the most common failure. Examples: "02_create_map failed!" #73 and "no valid gripper data" #74. Franka issues include #34 (flange settings, open) and #104 (Franka Hand). ([SLAM issues](https://github.com/real-stanford/universal_manipulation_interface/issues?q=is%3Aissue+slam), [Franka issues](https://github.com/real-stanford/universal_manipulation_interface/issues?q=is%3Aissue+franka))
- UMI hardware: a Google Doc hardware guide, YouTube assembly videos, and a BoM of $73 printed parts + $298 GoPro kit. ([GitHub](https://github.com/real-stanford/universal_manipulation_interface), [arXiv HTML](https://arxiv.org/html/2402.10329))
- UMI-FT setup pain points: same-subnet networking, a hardcoded iPhone IP, ONNX Runtime 1.16.3 in /opt, memory-hungry labelling, and UR-only interfaces. ([GitHub](https://github.com/real-stanford/UMI-FT))

#### ForceMimic, FoAR, ForceVLA, TA-VLA
- ForceMimic:
  - KeyError 'action_with_wrench' with the released hdf5; fixed by running the conversion script. ([hybridil #2](https://github.com/ForceMimic/hybridil/issues/2))
  - Unanswered UR camera-calibration question. ([hybridil #3](https://github.com/ForceMimic/hybridil/issues/3))
  - Gripper-version CAD missing, and the origin of the T265-to-F/T transforms is unclear. ([forcecapture issues](https://github.com/ForceMimic/forcecapture/issues))
  - The robot must carry the same tool as the capture device. ([forcecapture issues](https://github.com/ForceMimic/forcecapture/issues))
- FoAR:
  - An indentation bug froze the robot once force passed the threshold; confirmed by the maintainer. ([issue #6](https://github.com/Alan-Heoooh/FoAR/issues/6))
  - Users must implement eval_agent.py for their own hardware. ([DEPLOY.md](https://github.com/Alan-Heoooh/FoAR/blob/main/assets/docs/DEPLOY.md))
- ForceVLA:
  - forcevla_lora config not found and train.py arguments not recognised (#5).
  - "can't reproduce at all" (#6).
  - Unmapped submodules (#4, #7).
  - A community fork exists. ([issues](https://github.com/ft-robotic/ForceVLA/issues), [fork](https://github.com/nhathout/ForceVLA/tree/thesis-forcevla-edits))
- TA-VLA:
  - The loss implementation did not match the paper; later patched (#7).
  - With 100 clean demos on GR00T, TA-VLA scored 30% vs a 50% baseline; with 300 demos including recovery, 70% vs 60% (#13).
  - Models are not released (#2).
  ([issues](https://github.com/ZZongzheng0918/TA-VLA/issues/13))

#### FACTR and HIL-SERL
- FACTR:
  - The Franka follower node is not released and the ZMQ message flow is unclear. ([factr #1](https://github.com/RaindragonD/factr/issues/1), [factr_teleop #1](https://github.com/RaindragonD/factr_teleop/issues/1))
  - A user reaches about 330 Hz instead of 500 Hz on the Dynamixel loop. ([factr_teleop #7](https://github.com/RaindragonD/factr_teleop/issues/7))
- HIL-SERL ([issues](https://github.com/rail-berkeley/hil-serl/issues/45)):
  - ROS 2 port requested, no answer (#45).
  - jax 0.4.35 CUDA pinning (#47).
  - Rotation-representation and adjoint frame bugs (#41, #36, #38).
  - Replay-buffer RAM OOM (#26).
- serl_franka_controllers: franka_ros wants an RT kernel, which conflicts with CUDA on the same PC; the workaround is realtime_config: ignore. ([GitHub](https://github.com/rail-berkeley/serl_franka_controllers))

#### Licence status (GitHub API, 2026-11-02)
- MIT: RDP, ImplicitRDP (both confirmed from the LICENSE file text), hybridil and forcecapture, ACP, UMI-FT, OpenVLA-OFT, ForceFlow.
- Apache-2.0: ForceVLA, TA-VLA, Dream-Tac, FACTR, HIL-SERL, openpi.
- No licence: FoAR, how-to-peel, Feel the Force.
- Non-commercial: VT-Refine (NVIDIA licence), FlexiTac hardware (CC BY-NC 4.0).
- Sources: ([RDP LICENSE](https://raw.githubusercontent.com/xiaoxiaoxh/reactive_diffusion_policy/main/LICENSE), [ImplicitRDP LICENSE](https://raw.githubusercontent.com/Chen-Wendi/ImplicitRDP/main/LICENSE), [FoAR](https://github.com/Alan-Heoooh/FoAR), [how-to-peel](https://github.com/ToruOwO/how-to-peel), [Dream-Tac](https://github.com/LYFCLOUDFAN/Dream-Tac), [ForceFlow](https://github.com/JokerESC/ForceFlow))

#### Compute fit (stated figures)
- RDP: RTX 4090 inference with LDP at 100 ms; a user reports more than 20 GB VRAM for LDP training. ([arXiv HTML](https://arxiv.org/html/2503.02881v3), [issue #10](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/issues/10))
- FACTR: trains on one RTX 4090 in 2 to 6 h. ([arXiv HTML](https://arxiv.org/html/2502.17432v2))
- HIL-SERL: one RTX 4090. ([arXiv HTML](https://arxiv.org/html/2410.21845v3))
- UMI: single-GPU training tested on an RTX 3090 24 GB. ([GitHub](https://github.com/real-stanford/universal_manipulation_interface))
- openpi / ForceVLA: inference over 8 GB, LoRA over 22.5 GB. ([openpi](https://github.com/Physical-Intelligence/openpi))
- OpenVLA-OFT: training minimum 25.6 GB, inference 15.9 GB or more. ([project](https://openvla-oft.github.io/))
- Dream-Tac: H200 / A800 class. ([arXiv HTML](https://arxiv.org/html/2606.08737))
- FAWAM: 8x A100. ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
- ForceFlow: 4x RTX 4090, 8 to 10 h per task. ([arXiv HTML](https://arxiv.org/html/2605.11048))

### Inferences
- (inference) For a single student in 2 to 3 months on a Franka, the most realistic stack has three parts:
  1. Infrastructure: serl_franka_controllers or franky / libfranka Cartesian impedance at 1 kHz, from HIL-SERL/SERL (Franka-native, proven).
  2. Data collection with force feedback: FACTR's leader arm (Franka-native, Apache-2.0; FAWAM used it for cucumber peeling) or RDP's TactAR (Quest 3, MIT).
  3. Policy: RDP's codebase (MIT, peeling-proven), optionally with an ACP-style virtual-target + stiffness head.
- (inference) Expected time sinks: the RT-kernel vs CUDA split (two PCs), Polymetis vs deoxys vs franka_ros version conflicts, wrench frame and Jacobian correctness, interpolator rate, and calibration.
- (inference) Compute fit:
  - GTX 1650 4 GB: not enough to train any of these diffusion or VLA policies at paper settings. It might run small-policy inference (FACTR-size ACT, RDP's AT fast decoder) but was never tested by the authors.
  - pi0 / OpenVLA-OFT inference needs more than 8 GB and 15.9 GB respectively, so neither runs on the 1650.
  - Turing GPUs lack native bf16 (general knowledge, not from these sources), which the VLA codebases default to.
  - L4 24 GB: plausible for RDP, ACP, FACTR and HIL-SERL training and for pi0 / OpenVLA-OFT inference. pi0 LoRA is borderline (22.5 GB stated). OpenVLA-OFT training, Dream-Tac and FAWAM do not fit.
- (inference) Avoid basing the thesis on FoAR, Feel the Force or How to Peel code without written permission, because there is no licence. Avoid VT-Refine and FlexiTac for anything commercial (Vorwerk-type industry thesis) because of their non-commercial licences.

### Gaps
- No issue bodies were read for most UMI issues (titles only).
- The ACP optimizer bug (#4) is unconfirmed.
- No paper reports end-to-end person-months to reproduce.
- None of the GTX 1650 or L4 feasibility judgements were benchmarked.

## How much data did each need, and how was it collected?

### Takeaway
Peeling policies used 40 to 100 demonstrations per task (FAWAM 90 + 20 interventions; Dream-Tac 100; RDP 60; FoAR 50; How to Peel 50 cucumber; ForceVLA 43; ForceMimic 15 zucchinis split into 438 segments). VLA-class methods needed more (TA-VLA 400 per task; pi0 fine-tunes 5 to 100+ h). Collection with force feedback to the operator (haptic leader arm, AR force arrows, kinesthetic) is repeatedly shown to matter. Collection hours are almost never reported.

### Cited Findings
- RDP: 60 peeling demos via TactAR (Quest 3 AR force arrows). ([arXiv HTML](https://arxiv.org/html/2503.02881v3))
  - Demos collected in VR without tactile feedback lowered RDP (Force) peeling score by more than 30%.
  - TactAR user study: normalised peel length rose from 0.72 to 0.91, stable-contact ratio from 0.58 to 0.87.
- ForceMimic: about 5 min per zucchini with the handheld ForceCapture vs more than 13 min with force-feedback teleop; 15 zucchinis, 438 segments. ([arXiv](https://arxiv.org/abs/2410.07554), [arXiv HTML](https://arxiv.org/html/2410.07554v3))
- FoAR: 50 peeling demos by RH20T-style haptic teleop. ([arXiv HTML](https://arxiv.org/html/2411.15753v2))
- ForceVLA: Quest 3 VR teleop by 5 operators; 43 cucumber-peeling trajectories. ([arXiv HTML](https://arxiv.org/html/2505.22159v3), [HF API](https://huggingface.co/api/datasets/qiaojunyu/ForceVLA-real-data))
- FAWAM: 90 demos per task via a FACTR force-feedback leader, plus 20 human-intervention episodes per task for the residual corrector. ([arXiv HTML](https://arxiv.org/html/2606.08555v3))
- Dream-Tac: 100 demos per task by 3D SpaceMouse. ([arXiv HTML](https://arxiv.org/html/2606.08737))
- How to Peel: SpaceMouse teleop at 10 Hz; 50 cucumber, 150 apple, 200 potato demos, each one stroke. The paper also says 100 apple trajectories in another section. ([arXiv HTML](https://arxiv.org/html/2603.03280))
- TouchGuide: 50 cucumber-peeling demos with the handheld TacUMI (Vive tracker, about $720 base cost). ([arXiv HTML](https://arxiv.org/html/2601.20239))
- ACP: 200 wiping demos by kinesthetic teaching with haptic feedback. ([arXiv HTML](https://arxiv.org/html/2410.09309v2))
- UMI-FT: 275 wiping demos with a handheld iPhone + CoinFT gripper. ([arXiv HTML](https://arxiv.org/html/2601.09988v1))
- FACTR: 50 demos per task via a bilateral force-feedback leader arm (about $1,230). ([arXiv HTML](https://arxiv.org/html/2502.17432v2))
- HIL-SERL: 20 to 30 demos, then SpaceMouse interventions during 1 to 6 h of real-world RL; reward classifier from about 200 positive and 1000 negative frames. ([arXiv HTML](https://arxiv.org/html/2410.21845v3))
- TA-VLA: 400 demos per task. The author recommends about 50% recovery trajectories. ([arXiv HTML](https://arxiv.org/html/2509.07962v1), [issue #13](https://github.com/ZZongzheng0918/TA-VLA/issues/13))
- pi0: fine-tuning data of 5 h to 100+ h per task in the paper. ([arXiv HTML](https://arxiv.org/html/2410.24164))
- UMI: in-the-wild cup arrangement used 1400 demos in 12 person-hours. ([arXiv HTML](https://arxiv.org/html/2402.10329))
- ViTaMIn-B: experienced users collected 50 demos in 19.5 min. ([arXiv HTML](https://arxiv.org/html/2511.05858))

### Inferences
- (inference) A realistic peeling budget on a Franka is 60 to 100 demos per peeling condition, collected with operator force feedback (FACTR leader or TactAR). Add a short block of intervention or recovery episodes if a residual or RL stage is used.
- (inference) A handheld ForceCapture-style peeler with a wrist F/T is the fastest per-demo option (5 vs 13 min per zucchini). It forces the robot to carry the identical tool and sensor, and the reference design uses discontinued T265/L515 sensors.

### Gaps
- Total collection hours are not reported for RDP, FoAR, ForceVLA, FAWAM or Dream-Tac.
- No study compares FACTR-leader vs TactAR vs handheld collection on the same peeling task.

## Could any be trained partly in simulation, or are they real-data only?

### Takeaway
Every published peeling policy found is real-data only. The only visuo-tactile sim-fine-tuning method (VT-Refine) states that its simulator does not support deformable objects. Simulation is usable for Franka infrastructure and contact primitives: MuJoCo / robosuite wiping pretraining, the SERL and LeRobot gym_hil Franka sims, and LIBERO for VLA fine-tuning. Cutting has several dedicated simulators. No peeling simulator or benchmark was found.

### Cited Findings
- No simulation component in RDP, ForceMimic, FoAR, ForceVLA, FAWAM (none described), Dream-Tac (none mentioned) or How to Peel. ([RDP](https://arxiv.org/html/2503.02881v3), [ForceMimic](https://arxiv.org/html/2410.07554v3), [FoAR](https://arxiv.org/html/2411.15753v2), [ForceVLA](https://arxiv.org/html/2505.22159v3), [Dream-Tac](https://arxiv.org/html/2606.08737), [How to Peel](https://arxiv.org/html/2603.03280))
- VT-Refine: 30 real demos, then DPPO RL fine-tuning in Isaac Gym + TacSL. ([arXiv HTML](https://arxiv.org/html/2510.14930v1))
  - Visuo-tactile real success rose from 0.35-0.65 to 0.75-0.95 per asset.
  - "Isaac Gym and the tactile simulation do not support deformable objects".
  - Real-to-sim transfer cost about 5 to 10%.
- 3D-ViTac lists the lack of tactile simulation as a limitation. ([arXiv HTML](https://arxiv.org/html/2410.24091))
- TA-VLA: simulators such as LIBERO lack torque, so the work is real-world only. ([issue #12](https://github.com/ZZongzheng0918/TA-VLA/issues/12))
- Edinburgh/Sony few-shot tool use (arXiv 2507.13200, RA-L 2025 by DOI). ([arXiv HTML](https://arxiv.org/html/2507.13200))
  - Hardware: Franka Panda with tactile and proximity sensors.
  - Method: MuJoCo pretraining on 21 primitive surface trajectories, then 3 kinesthetic demos per tool.
  - Results: real wiped area 93.54% vs 72.74% demo-only.
- Adaptive Wiping (arXiv 2505.06451, UR5e): robosuite with 1000 domain-randomised samples pretrains a sponge VAE. Force tracking 96% vs admittance 42%. ([arXiv HTML](https://arxiv.org/html/2505.06451))
- Franka RL sims:
  - SERL MuJoCo franka_sim. ([sim quick start](https://github.com/rail-berkeley/serl/blob/main/docs/sim_quick_start.md))
  - LeRobot gym_hil MuJoCo Franka Panda (PandaPickCube) with intervention, Apache-2.0. ([hilserl_sim doc](https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl_sim.mdx), [gym-hil](https://github.com/huggingface/gym-hil))
- openpi supports LIBERO sim fine-tuning and evaluation. ([openpi](https://github.com/Physical-Intelligence/openpi))
- Cutting simulators:
  - DiSECt, differentiable FEM cutting. Background, 2021-2022. ([arXiv](https://arxiv.org/abs/2105.12244), [extended](https://arxiv.org/abs/2203.10263))
  - SliceIt!, real2sim2real knife slicing RL. ([arXiv](https://arxiv.org/abs/2404.02569))
  - TopoCut, a particle elastoplastic cutting benchmark. ([arXiv](https://arxiv.org/abs/2509.19712))
  - CulinaryCut-VLAP, an MLS-MPM cutting simulator with F/T labels. ([arXiv](https://arxiv.org/abs/2601.06451))
  - DexNinja, a differentiable sim coupling tactile contact and deformable cutting; sim augmentation adds more than 20%. ([arXiv](https://arxiv.org/abs/2610.17208))
  - These were checked at abstract level only.
- No robot peeling simulator was found. The closest is an orange-peel fracture simulation for material estimation, not control. ([arXiv](https://arxiv.org/abs/2606.16870))

### Inferences
- (inference) For cucumber peeling, simulation is best used for three things:
  1. Controller and safety development: impedance gains, reference limiting, stiffness-schedule sanity checks in MuJoCo with a rigid cylinder proxy.
  2. Pretraining the contact-tracking primitive (keep the tool normal to a curved surface at a target force), in the style of the Edinburgh/Sony wiping work.
  3. Testing the RDP / ACP pipeline end to end before the real FR3.
- (inference) Skin removal itself (blade engagement, chip formation) has no validated simulator. The final policy must be trained or fine-tuned on real demos.
- (inference) A sim-pretrain plus real fine-tune thesis is novel but risky. Building a peeling sim (for example an MPM cutting simulator adapted to shallow skin removal) is a research project of its own.

### Gaps
- No published sim-to-real result for peeling exists.
- Whether MuJoCo / MJX or Isaac Lab deformable FEM can model peeler-skin interaction well enough was not investigated (outside this note's scope; the sim-to-real notes may cover it).
- The cutting simulator details are abstract-level only.
