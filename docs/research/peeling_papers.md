# Robotic peeling of produce with robot arms: paper catalogue (cucumber first), checked 2026-11-02

Scope notes:
- Venue labels: "VERIFIED" means checked against the proceedings or official venue page. "per citation" means taken from another paper's reference list or an author page, not from the proceedings. "arXiv-only" means no venue found.
- "unverified" marks anything not confirmed in a primary source.
- Numbers are copied from the papers. Nothing is estimated.
- Some items come from four helper agents that read the arXiv HTML or PDFs directly. Their source URLs are kept inline.

## Q1. Which papers do cucumber peeling, with what exact setup and success numbers? (Includes the catalogue of all produce-peeling papers.)

### Takeaway
At least 20 works peel a cucumber with a robot, most from 2024-2026.
- The strongest learned results come from force- or tactile-aware imitation learning: RDP (RSS 2025), FoAR (RA-L 2025), ForceVLA (NeurIPS 2025), TouchGuide (RSS 2026), and the Berkeley knife-peeling paper (2026).
- Almost all of them run on Flexiv Rizon, xArm, ARX, UR or Kinova arms.
- Only two cucumber works run on a Franka. MORPHeus (Panda, ICRA 2024) uses an impedance controller and a fixture. Any-ttach (FR3, RSS 2026 workshop) uses a Diffusion Policy with a mounted peeler.
- None of them simulates the peeling itself.

### Cited Findings

#### A. Cucumber peeling papers (main catalogue)

**A1. Reactive Diffusion Policy (RDP): Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation**
- **Who:** Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, Cewu Lu (SJTU / Tsinghua).
- **Venue:** RSS 2025, VERIFIED in the proceedings - [RSS XXI p052](https://www.roboticsproceedings.org/rss21/p052.html).
  - Listed as a best student paper finalist on Huazhe Xu's homepage - [homepage source](https://github.com/xuhuazhe/xuhuazhe.github.io/blob/master/index.html).
- **Robot and sensors:** two Flexiv Rizon 4 arms with joint torque sensors and Flexiv Grav grippers. Peeling uses one arm with one wrist RealSense D435.
  - Tactile/force variants: GelSight Mini (25 FPS), an improved MCTac, or the Flexiv RDK TCP wrench estimated from joint torques (120 Hz, downsampled to 24 FPS) - [arXiv 2503.02881](https://arxiv.org/html/2503.02881v3).
- **Task:** "The robot needs to grasp the peeler, approach a cucumber held midair by a human hand, then begin peeling." The peeler is grasped by the gripper at varying grasp locations, and the cucumber is held by a human - [arXiv](https://arxiv.org/html/2503.02881v3).
- **Method:** a slow latent diffusion policy (12 FPS action chunks) plus a fast asymmetric-tokenizer policy (24 FPS) that reacts to tactile/force signals. Demos come from TactAR, an AR teleop system on a Quest 3 that shows tactile/force feedback - [arXiv](https://arxiv.org/html/2503.02881v3).
- **Data and evaluation:** 60 demos are referenced for Peeling in the data-quality study. 10 trials for each of 3 variations: no perturbation, perturbation before contact, perturbation after contact - [arXiv](https://arxiv.org/html/2503.02881v3).
- **Metric:** proportion of peeled skin over cucumber length, normalized by the demo average - [project page](https://reactive-diffusion-policy.github.io/).
- **Results (score, overall):**
  - DP 0.44
  - DP with tactile image 0.41
  - DP with tactile embedding 0.39
  - RDP (GelSight) 0.90
  - RDP (MCTac) 0.88
  - RDP (Force) 0.95
  - Per condition, RDP (Force) scores 0.99 / 0.98 / 0.88 against DP's 0.56 / 0.58 / 0.19 - [arXiv](https://arxiv.org/html/2503.02881v3).
- **Data quality:**
  - Tactile/force feedback in TactAR raised teleop normalized peel length from 0.72 to 0.91, and the stable-contact-force ratio from 0.58 to 0.87 (10 users).
  - Training on 60 VR demos collected without feedback dropped RDP (force) by more than 30% - [arXiv](https://arxiv.org/html/2503.02881v3).
- **Stated limitations:** TactAR is less intuitive than direct hand operation, and the system is designed for two-finger grippers only - [arXiv](https://arxiv.org/html/2503.02881v3).
- **Code:** project page - [reactive-diffusion-policy.github.io](https://reactive-diffusion-policy.github.io/). Whether code is released was not checked (unverified).

**A2. FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation**
- **Who:** Z. He, H. Fang, J. Chen, H. Fang, C. Lu (SJTU).
- **Venue:** IEEE RA-L 2025, per citation in [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1). Not checked on IEEE Xplore.
- **Robot and sensors:** Flexiv Rizon, Dahuan AG-95 gripper, OptoForce F/T sensor between flange and gripper, front RealSense D435 - [arXiv 2411.15753](https://arxiv.org/html/2411.15753v2).
- **Control:** plain end-effector position control. "No advanced control strategies like compliance control, admittance control, and hybrid force/torque control are used" - [arXiv](https://arxiv.org/html/2411.15753v2).
- **Task and data:** cucumber peeling with a grasped peeler, with the grasp position on the tool varied. 50 demos via haptic teleop, 20 trials per method - [arXiv](https://arxiv.org/html/2411.15753v2).
- **Metric:** peeled length proportion, normalized by the demo average of 0.778 - [arXiv](https://arxiv.org/html/2411.15753v2).
- **Results (score / grasp % / peel %):**
  - ACT 0.120 / 95 / 25
  - DP 0.386 / 85 / 70
  - RISE 0.377 / 100 / 50
  - RISE force-concat 0.524 / 100 / 80
  - FoAR 0.756 / 100 / 100
  - Source: [arXiv](https://arxiv.org/html/2411.15753v2).
- **Reactive control thresholds:** contact probability 0.9, force 8 N, step 6 mm - [arXiv](https://arxiv.org/html/2411.15753v2).
- **Project page:** [tonyfang.net/FoAR](https://tonyfang.net/FoAR/). How the cucumber is fixed is unverified.

**A3. ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation**
- **Venue:** NeurIPS 2025, VERIFIED - [NeurIPS poster 120169](https://neurips.cc/virtual/2025/poster/120169), [OpenReview PDF](https://openreview.net/pdf?id=2845H8Ua5D).
- **Robot and sensors:** Flexiv Rizon 7-DoF, Dahuan adaptive gripper, RealSense D435 third-person and D415 wrist cameras. The 6-D F/T input source is not stated in the passages read (unverified). Quest 3 VR teleop - [arXiv 2505.22159](https://arxiv.org/html/2505.22159).
- **Method:** pi0-based VLA with a Force-aware Mixture-of-Experts (FVLMoE) fusing force after the VLM.
- **Data and evaluation:** about 50 demos per task. Cucumber peeling has 15 trials, each with 15 peeling strokes - [arXiv](https://arxiv.org/html/2505.22159).
- **Cucumber results:**

| Model | Avg peel length per stroke | Min strokes to clean |
| --- | --- | --- |
| pi0-base without force | 10.27 cm | 14 |
| pi0-base with force | 13.17 cm | 10 |
| ForceVLA | 14.12 cm | 7 |

  - Source: [arXiv](https://arxiv.org/html/2505.22159).
- **Data release:** ForceVLA-Data (244 trajectories over 5 tasks) "will be made publicly available" - [project site](https://sites.google.com/view/forcevla2025).
- How the cucumber is held is unverified.

**A4. How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference**
- **Who:** Toru Lin, Shuying Deng, Zhao-Heng Yin, Pieter Abbeel, Jitendra Malik (UC Berkeley).
- **Venue:** arXiv 2603.03280 (Mar 2026), with an OpenReview forum entry [DzAZfcPUiP](https://openreview.net/forum?id=DzAZfcPUiP). The venue could not be read (unverified).
- **Links:** [project](https://toruowo.github.io/peel/), [GitHub ToruOwO/how-to-peel](https://github.com/ToruOwO/how-to-peel), [arXiv](https://arxiv.org/html/2603.03280v1).
- **Hardware:** Kinova Gen3 7-DoF, torque-controlled. Impedance control runs at 500 Hz with 10 Hz policy commands. ATI Mini45 F/T at 500 Hz, custom knife mount, two D405 wrist cameras - [arXiv](https://arxiv.org/html/2603.03280v1).
- **Tool:** a knife, not a peeler. How the produce is fixed is not stated on the page or in the passages read (unverified).
- **Data collection:** SpaceMouse teleop - [arXiv](https://arxiv.org/html/2603.03280v1).
  - Teleop methods compared on 10 cucumber trajectories:
    - heuristic point-cloud planner 20% (score 1.8)
    - VR 20% (2.1)
    - SpaceMouse 100% (8.5)
    - kinesthetic teaching 100% raw, but 0% on compliant replay
- **Demos:** 50 cucumber, 150 apple, 200 potato. Each demo is one stroke along the principal axis.
- **Policy:** diffusion policy over grayscale RGB-D, F/T and joints, then preference-based residual finetuning with a learned reward.
- **Success definition:** human score above 3.
- **Results** - [arXiv](https://arxiv.org/html/2603.03280v1):
  - Cucumber: 50 trajectories (about 8 cucumbers) give 100% success, average score 8.5 (10 trials).
  - All in-category tests: 100%.
  - Zero-shot to new produce: cucumber policy on zucchini 50%, apple policy on pear 90%, potato policy on daikon 80%.
  - Ablations on potato:
    - RGB instead of grayscale: 0%
    - F/T only: 0%
    - 40 trajectories: 10%
    - 200 trajectories: 80% before finetuning
- **Internal inconsistency:** the abstract says over 90% on out-of-distribution produce, while the body says "over 70% average success rates on unseen produce" - [arXiv](https://arxiv.org/html/2603.03280v1).
- **Stated limitations:** depends on human teleop demos, perception is a bottleneck (suggests a front camera), and food waste. They hope for "surrogate produce" or improved sim-to-real - [arXiv](https://arxiv.org/html/2603.03280v1).

**A5. TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance**
- **Who:** Zhemeng Zhang et al., corresponding author Daolin Ma (SJTU, Xense Robotics, Shanghai AI Lab, Oxford, UCSD and others).
- **Venue:** RSS 2026, VERIFIED - [RSS XXII p078](https://www.roboticsproceedings.org/rss22/p078.html); [arXiv 2601.20239](https://arxiv.org/abs/2601.20239).
- **Robot and sensors:** peeling runs on the Bi-ARX5 dual-arm robot with Xense tactile sensors on the outer fingers, plus head and wrist cameras.
- **Setup:** the right gripper picks up the peeler. The left arm grasps a randomly placed cucumber (yaw about ±30 deg).
- **Method:** a frozen DP or pi0.5 base policy, steered at inference time by a contrastively trained Contact Physical Model. 50 demos via the TacUMI handheld device, 20 trials per method - [RSS PDF](https://www.roboticsproceedings.org/rss22/p078.pdf).
- **Metric:** robot peel length divided by expert peel length, rounded to {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1}.
- **Results (score):**
  - DP 0.500
  - RDP 0.740
  - TouchGuide on DP (tactile image) 0.810
  - pi0.5 0.785
  - TouchGuide on pi0.5 (tactile image) 0.975; (force) 0.955
  - Source: [RSS PDF](https://www.roboticsproceedings.org/rss22/p078.pdf).

**A6. Learning High-Frequency Continuous Action Chunks in Latent Space**
- **Who:** Kunyun Wang, Yuhang Zheng, Yupeng Zheng, Jieru Zhao, Wenchao Ding.
- **Venue:** ICML 2026, VERIFIED - [OpenReview aphjc9yoLf](https://openreview.net/forum?id=aphjc9yoLf); [arXiv 2605.24931](https://arxiv.org/abs/2605.24931).
- **Robot and sensors:** single xArm 7 with a Robotiq 2F-85. No F/T or tactile sensing mentioned.
- **Task and metric:** "Peel Cucumber". Success means more than half the skin peeled. 50 real trials per method.
- **Results, asynchronous inference (original -> latent + RTR):**
  - DP 82% -> 90%
  - OpenVLA-OFT 20% -> 70%
  - pi0.5 72% -> 80%
  - Source: [OpenReview PDF](https://openreview.net/pdf?id=aphjc9yoLf).
- **Code:** [github.com/tars-robotics/RTR](https://github.com/tars-robotics/RTR).

**A7. Interaction Latents as Actions for Contact-Rich Manipulation (ILA)**
- **Who:** Yuhang Zheng et al.
- **Venue:** arXiv 2610.18855 (Oct 2026), arXiv-only - [arXiv](https://arxiv.org/abs/2610.18855).
- **Robot and sensors:** dual-arm ARX X5, 4 fingertip Xense tactile sensors, D405 wrist camera plus an external camera.
- **Data:** 150 demos per task, collected by physically guiding the arms.
- **Success definition:** 3 peel-and-place cycles of complete, thick strips. 20 rollouts per method.
- **Results:**
  - FM-tac 50%
  - FM-ILA 75%
  - VLA-ILA 50%
  - WAM-ILA 70%
  - FM-ILA plus their offline improvement 85%
  - Replay of recorded actions 40%
  - Source: [arXiv HTML](https://arxiv.org/html/2610.18855v2).
- **Related project page:** the Phoenix page [openphoenix-team.github.io](https://openphoenix-team.github.io/Phoenix-pages/) shows different peeling numbers for overlapping methods (FastWAM 10% vs 70% in the paper, with the same Fext L1 of 2.01). Treat the page numbers as unverified.

**A8. OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation**
- **Venue:** arXiv 2603.19201 (v3), arXiv-only - [arXiv](https://arxiv.org/html/2603.19201).
- **Robot and sensors:** xArm7 with fingertip tactile sensors and a wrist D435. Data from the TacUMI handheld device plus robot.
- **Produce:**
  - Training: cucumber, Chinese yam, radish, carrot, daikon.
  - Evaluation: cucumber, yam, radish.
  - 150 trajectories per object.
- **Setup:** the object is "placed at a random position and fixed during manipulation". 10 trials per object - [arXiv](https://arxiv.org/html/2603.19201).
- **Peel success rate (object diversity / height generalization / perturbation):**
  - DP 0.06 / 0 / 0
  - ForceMimic 0.27 / 0.18 / 0
  - RDP 0.48 / 0.36 / 0.45
  - OmniVTA 0.55 / 0.48 / 0.63
  - Source: [arXiv](https://arxiv.org/html/2603.19201).
- **Observed baseline failure:** RDP's reactive policy "often produces overly strong contacts that damage the sensors" - [arXiv](https://arxiv.org/html/2603.19201).

**A9. TAO-Force: Unifying Force-Aware Perception and Fast-Slow Control for Contact-Rich Manipulation**
- **Venue:** arXiv 2609.18497 (Sep 2026), arXiv-only - [arXiv](https://arxiv.org/html/2609.18497).
- **Robot and sensors:** AgiBot G1 dual-arm humanoid with 6-axis F/T at each wrist.
- **Control:** a VLA with F-FiLM force conditioning. Admittance control at 1 kHz takes over on contact.
- **Success definition:** a continuous peel longer than half the cucumber's length. 30 trials per setting.
- **Results (in-distribution / table lowered 3 cm):**
  - GR00T N1.7: 76.7 / 26.7
  - pi0.5: 53.3 / 53.3
  - TAO-Force: 96.7 / 86.7
  - Source: [arXiv](https://arxiv.org/html/2609.18497).

**A10. FORCA: A Force-Aware Coding Agent for Contact-Rich Robot Manipulation**
- **Who:** Xiao Tan, Yixuan Wang, Zhexi Wang, Wenzhen Yuan (UIUC).
- **Venue:** arXiv 2610.31841 (Oct 2026), arXiv-only - [arXiv](https://arxiv.org/html/2610.31841).
- **Robot and sensors:** UR5e with a wrist F/T sensor at 100 Hz.
- **Setup:** the cucumber is held in a person's hand.
- **Method:** an LLM agent writes force-feedback Python code. No demos, no training, no simulation.
- **Results (10 trials; contact loss / excessive force / peeling ratio, %):**
  - Vision-only agent: 40.6 / 13.4 / 37.4
  - FORCA: 2.2 / 2.7 / 81.1
  - Source: [arXiv](https://arxiv.org/html/2610.31841).

**A11. Any-ttach: Quick End-effector Swapping Enables Manipulation Dexterity with Simplicity**
- **Venue:** RSS 2026 dexterous manipulation workshop, anonymous - [PDF](https://dex-manipulation.github.io/rss2026/pdfs/8_Any_ttach_Quick_End_effector.pdf), [project](https://any-ttach.github.io/).
- **Robot:** Franka Research 3 (FR3). The peeler is mechanically locked into a quick-swap coupling on the gripper, not grasped. No F/T or tactile sensing.
- **Method:** Diffusion Policy, 30 demos per skill. Contact is approximated by a 5-7 mm positional offset along the tool normal (impedance heuristic).
- **Success definition:** one continuous stroke end to end.
- **Results:**
  - Peel 9/15.
  - Grasp-based peeler holding "could not be demonstrated".
  - Cut 14/15, spear 8/15.
  - Source: [PDF](https://dex-manipulation.github.io/rss2026/pdfs/8_Any_ttach_Quick_End_effector.pdf).

**A12. MORPHeus: a Multimodal One-armed Robot-assisted Peeling System with Human Users In-the-loop**
- **Who:** R. Ye, Y. Hu, Y. A. Bian, L. Kulm, T. Bhattacharjee (Cornell EmPRISE).
- **Venue:** ICRA 2024 pp. 9540-9547, per citation in [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1); [arXiv 2404.06570](https://arxiv.org/html/2404.06570); [project](https://emprise.cs.cornell.edu/morpheus/).
- **Robot and fixture:** Franka Emika Panda. An assistive cutting board with a moving jaw holds the food, and the board is fixed to the table with suction cups. A 3D-printed peeler holder is fixed to the table - [arXiv](https://arxiv.org/html/2404.06570).
- **Sensors:** the peeler carries visual, force and vibration sensing. An LSTM classifies peeled vs unpeeled areas.
- **Planning:** GPT-4 generates PDDL plans with the user in the loop.
- **Control:** Cartesian impedance - [arXiv](https://arxiv.org/html/2404.06570).
  - Three gain sets: kt = 120 / 150 / 180, kr = 0 / 5 / 10.
- **Peel success per food:**
  - Evaluation: 10 trials per food. Success means "skin is separated from the flesh completely", with manual rotation between strokes.
  - Cucumber: 2/10 (low stiffness), 10/10 (medium), 10/10 (high).
  - Potato: 0/10, 2/10, 10/10.
  - Acorn squash best: 5/10. Radish best: 3/10.
  - Perception on cucumber: 7/10 with the peel action, 9/10 with the slide action.
  - Source: [arXiv](https://arxiv.org/html/2404.06570).
- **Foods tested (12):** cucumber, carrot, Chinese yam, Chinese okra, Chinese eggplant, potato, zucchini, gold bar squash, daikon, apple, radish, acorn squash - [arXiv](https://arxiv.org/html/2404.06570).

**A13. Learning Multimodal Contact-Rich Skills from Demonstrations Without Reward Engineering**
- **Who:** Balakuntala, Kaur, Ma, Wachs, Voyles (Purdue).
- **Venue:** ICRA 2021, per the arXiv comment - [arXiv 2103.01296](https://arxiv.org/html/2103.01296v2).
- **Robot and sensors:** Rethink Sawyer in impedance mode, Barrett BH-282 hand with tactile sensors, 6-DoF wrist F/T, RealSense D435i.
- **Data and policy:** 30 demos of peeling one side of a cucumber. SAC policy, about 30 min of real-world training.
- **Success definition:** peeled/unpeeled area ratio within 10% of the demo reference.
- **Results:** proposed 100%, modeled controller 65%, naive RL 15% - [arXiv](https://arxiv.org/html/2103.01296v2).

**A14. Self-Evaluation in One-Shot Learning from Demonstration of Contact-Intensive Tasks**
- **Who:** Balakuntala et al. (Purdue).
- **Venue:** arXiv 1904.01846; venue and robot unverified.
- **Task:** cucumber peeling from one demo, with Q-learning self-evaluation.
- **Details:** minimum contact force 0.5 N, +0.3 N per corrective action. "Peeled" means at least 10% of the visible surface - [arXiv](https://arxiv.org/html/1904.01846).

**A15. Robotic Hand Tool Use with Contact-Based Demonstration: The Case of Cucumber Peeling**
- **Who:** Xie et al. (Tencent Robotics X / SCUT).
- **Venue:** IROS 2025 pp. 10386-10393, DOI 10.1109/IROS60139.2025.11247573 - [Semantic Scholar](https://www.semanticscholar.org/paper/3c0321b2ec6263bd4c5e0fc9da731cbcce5dbe06), [dblp](https://dblp.org/pid/96/928).
- **Setup:** a three-fingered hand holds a peeler.
- **Method:** a handheld teaching device with optical tactile sensors; object-level stiffness control.
- **Numbers:** none in the abstract (unverified).

**A16. Robotic Hand-Arm Coordinated Skill Learning for Tool Manipulation Tasks** (journal follow-up to A15)
- **Venue:** CAAI Transactions on Intelligence Technology, online Oct 2026, DOI 10.1049/cit2.70185 - [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70185).
- **Robot and sensors:** ABB IRB1100 with the TRX-Hand3 (8 DoF). The peeler is clamped in the hand and the cucumber sits in a fixture. TacTip-style teaching device; ATI Mini-45 used for calibration.
- **Method:** multimodal Diffusion Policy.
- **Result:** on a tail-curved cucumber, peel length was 3-4 cm without stiffness control and 6.5-7.5 cm with it, against a 7 cm target - [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70185).

**A17. Bimanual Tactile-Augmented Teleoperation for Contact-Rich Robotic Manipulation: A Pilot Evaluation (BiTAT)**
- **Who:** Shi, Bao, Zheng, Zhao (Zhejiang Univ.).
- **Venue:** MDPI Applied Sciences 16(17):8383, 2026 - [MDPI](https://www.mdpi.com/2076-3417/16/17/8383).
- **Robot and sensors:** two xArm6 arms with capacitive tactile grippers.
- **Setup:** one gripper holds the cucumber and the other holds the peeler.
- **Success definition:** a continuous strip longer than 8 cm.
- **Diffusion Policy results (20 demos / 50 demos):**
  - Vision only: 15% / 60%
  - Plus tactile embedding and Transformer: 40% / 85% (84.0 ± 4.2% over 5 seeds)
  - Source: [MDPI](https://www.mdpi.com/2076-3417/16/17/8383).

**A18. Design and Experimental Validation of Sensorless 4-Channel Bilateral Teleoperation for Low-Cost Manipulators**
- **Venue:** arXiv 2507.06174 (v7); venue and authors unverified.
- **Robot:** dual CRANE-X7 arms, no force sensors (force from a disturbance observer), D435i camera.
- **Setup:** the left arm grasps a cucumber on a cutting board; the right arm grabs the peeler.
- **Data and policy:** 10 demos, ACT.
- **Peel success:**
  - No force in input or output: 0/5
  - Force as input: 3/5
  - Force as input and output: 2/5
  - Source: [arXiv](https://arxiv.org/html/2507.06174).

**A19. Object-centric Task Representation and Transfer using Diffused Orientation Fields**
- **Who:** Cem Bilaloglu, Tobias Löw, Sylvain Calinon (Idiap / EPFL).
- **Venue:** Science Robotics 2026, DOI 10.1126/scirobotics.aea1762 - [Science](https://www.science.org/doi/10.1126/scirobotics.aea1762). The PDF read was the arXiv 2511.18563 version - [Calinon PDF](https://calinon.ch/papers/Bilaloglu-ScienceRobotics2026.pdf). The journal version may differ.
- **Robot and sensors:** uFactory Lite 6, BotaSys SensONE F/T, D405 camera.
- **Setup:** the peeler is in a 3D-printed mount and the object is in a fixture. Admittance control.
- **Produce:** pear, avocado, cucumber.
- **Results:** no learning for peeling and no success rate reported. Force profiles are shown in figures.
- **Code:** [github.com/idiap/diffused_fields_robotics](https://github.com/idiap/diffused_fields_robotics).

**A20. pi0.7 technical report (Physical Intelligence, Apr 2026)**
- **Task:** "Peel Fruits and Vegetables" on zucchini, cucumbers and carrots. One gripper pins the item against a cutting board and the other peels - [pi.website/pi07](https://www.pi.website/pi07).
- **Metric:** a 9-point rubric (peel quartiles plus cleanup).
- **Results:** shown only as a bar chart against a pi0.6 specialist; exact numbers unverified.
- **Robot:** platform not identified (unverified). Not peer reviewed.

#### B. Non-cucumber produce peeling papers

**B1. ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System for Contact-Rich Manipulation (zucchini)**
- **Who:** Wenhai Liu, Junbo Wang, Yiming Wang, Weiming Wang, Cewu Lu (SJTU).
- **Venue:** ICRA 2025 pp. 1105-1112, per citation in [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1); [arXiv 2410.07554](https://arxiv.org/html/2410.07554).
- **Robot and setup:** two Flexiv Rizon 4 arms - [arXiv](https://arxiv.org/html/2410.07554).
  - The left arm's gripper holds the zucchini (rule-based stabilization).
  - The right arm has a fixed peeler (the same tool as on the capture device).
  - External L515 RGB-D camera; point-cloud input.
- **Data:** the robot-free ForceCapture handheld (6-axis F/T, T265 SLAM). 15 zucchinis gave 438 peeling segments and 30,199 action sequences.
- **Policy:** HybridIL predicts wrench plus pose. When predicted force stays above 6 N, it switches to Flexiv RDK hybrid force-position control - [arXiv](https://arxiv.org/html/2410.07554).
- **Results (success %, "motion correct" / "peel longer than 10 cm"):**
  - Raw DP: 80 / 55 (20 trials)
  - Force DP: 60 / 10 (10 trials)
  - Force+Hybrid DP: 80 / 20 (10 trials)
  - HybridIL: 100 / 85 (20 trials)
  - Source: [arXiv](https://arxiv.org/html/2410.07554).
- **Collection speed:** peeling a zucchini took about 5 min with ForceCapture vs more than 13 min by force-feedback teleop.
- **Hardware, code and data:** at [forcemimic.github.io](https://forcemimic.github.io), per the abstract.

**B2. Vegetable Peeling: A Case Study in Constrained Dexterous Manipulation**
- **Who:** Tao Chen, Eric Cousineau, Naveen Kuppuswamy, Pulkit Agrawal (MIT / TRI).
- **Venue:**
  - CoRL 2023 Workshop TGR poster - [OpenReview submissions list](https://openreview.net/submissions?venue=robot-learning.org/CoRL/2023/Workshop/TGR).
  - Listed as ICRA 2025 on the author's homepage - [taochenshh.github.io](https://taochenshh.github.io/). ICRA proceedings not checked.
  - [arXiv 2407.07884](https://arxiv.org/html/2407.07884v1).
- **Robot:**
  - An Allegro hand (a LEAP hand was also tested) on one Franka Panda reorients the produce in hand.
  - A second Franka Panda grasps a peeler in its gripper - [arXiv](https://arxiv.org/html/2407.07884v1).
- **Peeling control:** done in one of two ways - [arXiv](https://arxiv.org/html/2407.07884v1):
  - Teleop: Haption Virtuose 6D, 200 Hz operational-space impedance in Drake.
  - Vision script: Grounded SAM, then point cloud, then PCA long axis, then a 2 cm slice, then a spline, then Cartesian position control with a fixed peeler orientation.
- **Reorientation results** on pumpkin, melon, radish, papaya - [arXiv](https://arxiv.org/html/2407.07884v1):
  - 90% of tests rotated the surface at least 4 cm in 3.5 s and at least 7.3 cm in 7 s.
  - Stops in 0.4 s on average.
  - Firm grasp in 90% of tests.
- **Peel outcome:** reported only qualitatively ("in most cases" the grasp held).

**B3. Goal-conditioned dual-action imitation learning for dexterous dual-arm robot manipulation (banana)**
- **Who:** Kim, Ohmura, Kuniyoshi (Univ. Tokyo).
- **Venue:** IEEE T-RO 2024, per citation in [arXiv 2404.06570](https://arxiv.org/html/2404.06570); [arXiv 2203.09749](https://arxiv.org/html/2203.09749).
- **Robot and sensors:** dual UR5 arms with 1-DoF grippers, gaze-foveated images.
- **Results (consecutive peeling on 22 bananas):** full-task success 40.91%. PeelNeck 95.45%, PeelRight 59.09%, PeelLeft 63.63% - [arXiv](https://arxiv.org/html/2203.09749).
- **Caveat:** banana necks were pre-cut to 1 cm depth.

**B4. Other produce in the papers above:** apple and potato with a knife (A4); zucchini, carrot, potato, apple, daikon, squashes, eggplant (A12); zucchini and carrot (A20); pear and avocado (A19); yam, radish, carrot, daikon (A8).

#### C. Checked and found NOT to contain produce peeling (to show coverage)
Source for each: [helper read of arXiv HTML](https://arxiv.org/html/2512.10946).
- ImplicitRDP 2512.10946: names fruit peeling only as an example task type.
- ForceVLA2 2603.15169
- UMI-FT 2601.09988: zucchini skewering, not peeling.
- Force Policy 2602.22088
- PhaForce 2603.08342
- ContactBench 1.0 2610.24512: the 6 evaluated tasks only.
- DEFT-VLA 2610.14854
- ForceDelta-VLA 2609.18242: Franka Panda.
- Wrench-ACT 2609.37552
- Imp-ACT 2609.31225: Franka Panda + ATI.
- FP2 2609.37433
- ForeTac 2610.18118
- TacThru 2512.09851
- exUMI 2509.14688
- Touch in the Wild 2507.15062
- Feel the Force 2506.01944
- ACP 2410.09309
- DexForce 2501.10356
- TA-VLA 2509.07962
- FreeTacMan 2506.01941
- N0-TWAM 2607.23783
- FARM 2510.13324: Franka FR3.
- TEAR 2610.30819: no peeling despite a social-media mention.

### Inferences
- Cucumber peeling has become a de facto benchmark for force/tactile-aware imitation learning since 2025. RDP, FoAR, TouchGuide and OmniVTA all report it, and later papers use RDP as a baseline (TouchGuide, OmniVTA).
- Metrics are not comparable across papers:
  - Normalized peel length: RDP, FoAR.
  - Strip-length thresholds: ForceMimic over 10 cm, BiTAT over 8 cm, TAO-Force over half the length.
  - Peel per stroke: ForceVLA.
  - Human score: knife paper.
  - Strict cycles: ILA.
- A Franka reproduction should report at least normalized peel length plus a strip-length threshold, so results can be compared with both groups.
- The highest robustness numbers come from closing a fast force loop around a learned slow policy (RDP force 0.95; TAO-Force 96.7%) or from explicit hybrid force control (ForceMimic 100/85). Pure vision policies stay around 0.4-0.6 normalized score.

### Gaps
- No mango peeling paper was found. No carrot-specific learned paper was found beyond the multi-food systems (A8, A12, A20).
- Google Scholar citation counts were not collected.
- How the cucumber is held is unverified for FoAR, ForceVLA, the knife paper (produce fixture) and ILA.
- The F/T source in ForceVLA is unverified.
- The pi0.7 peeling numbers are in a chart only.
- Not checked against IEEE Xplore: the venues of FoAR, ForceMimic and MORPHeus (taken from citations), and the ICRA 2025 listing for B2.

## Q2. Which peeling papers use a Franka, and which use other arms that could be swapped for a Franka?

### Takeaway
Franka-based peeling work is thin:
- MORPHeus: Panda, Cartesian impedance, a fixture, an instrumented peeler.
- Tao Chen et al.: two Pandas, peeling by teleop or a vision script.
- Any-ttach: FR3, Diffusion Policy, mounted peeler, 9/15.

The best learned cucumber results use Flexiv Rizon arms, whose joint-torque sensing and built-in hybrid force control are the closest analogue to a Franka. Kinova Gen3, xArm and UR5e setups are also portable.

### Cited Findings
- Franka-based:
  - MORPHeus, Panda - [arXiv](https://arxiv.org/html/2404.06570).
  - Vegetable Peeling, two Pandas, Allegro on one - [arXiv](https://arxiv.org/html/2407.07884v1).
  - Any-ttach, FR3 - [PDF](https://dex-manipulation.github.io/rss2026/pdfs/8_Any_ttach_Quick_End_effector.pdf).
  - DiSECt cutting (not peeling): Franka Panda with a knife cutting cucumber/apple/banana, with force from Panda joint torques - [arXiv 2203.10263](https://arxiv.org/abs/2203.10263).
- Flexiv Rizon:
  - RDP: Rizon 4, joint-torque TCP wrench via RDK - [arXiv](https://arxiv.org/html/2503.02881v3).
  - ForceMimic: Rizon 4 x2, RDK hybrid force-position primitives - [arXiv](https://arxiv.org/html/2410.07554).
  - FoAR: Rizon + OptoForce - [arXiv](https://arxiv.org/html/2411.15753v2).
  - ForceVLA: Rizon 7-DoF - [arXiv](https://arxiv.org/html/2505.22159).
- Kinova Gen3: knife paper, impedance control with an ATI Mini45 - [arXiv](https://arxiv.org/html/2603.03280v1).
- xArm7: OmniVTA - [arXiv](https://arxiv.org/html/2603.19201); ICML 2026 latent-chunk paper - [OpenReview](https://openreview.net/pdf?id=aphjc9yoLf).
- xArm6 x2: BiTAT - [MDPI](https://www.mdpi.com/2076-3417/16/17/8383).
- UR5e: FORCA - [arXiv](https://arxiv.org/html/2610.31841).
- Dual UR5: Kim banana - [arXiv](https://arxiv.org/html/2203.09749).
- Sawyer: Balakuntala 2021 - [arXiv](https://arxiv.org/html/2103.01296v2).
- uFactory Lite 6: Bilaloglu - [Calinon PDF](https://calinon.ch/papers/Bilaloglu-ScienceRobotics2026.pdf).
- ARX dual arms: TouchGuide - [RSS PDF](https://www.roboticsproceedings.org/rss22/p078.pdf); ILA - [arXiv](https://arxiv.org/html/2610.18855v2).
- AgiBot G1: TAO-Force - [arXiv](https://arxiv.org/html/2609.18497).
- CRANE-X7: sensorless bilateral - [arXiv](https://arxiv.org/html/2507.06174).
- ABB IRB1100 + TRX-Hand3 - [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70185).
- Older systems:
  - PR2 grapefruit (cutting, not peeling) - [arXiv 1909.07247](https://arxiv.org/html/1909.07247).
  - JSK life-sized dual-arm humanoid, sweet potato - [Paladyn 2013](https://www.degruyterbrill.com/document/doi/10.2478/pjbr-2013-0013/html).

### Inferences
- Easiest ports to an FR3:
  - RDP (force variant). It uses a wrench estimated from joint torques, which an FR3 also provides as an external-wrench estimate.
  - FoAR. It only needs position control plus an F/T stream. An FR3 would need its own wrist F/T or the torque-based estimate.
  - MORPHeus. Already Panda, with published impedance gains.
  - Any-ttach. Already FR3.
- Tool holding matters:
  - Any-ttach reports grasp-based peeler holding "could not be demonstrated" - [PDF](https://dex-manipulation.github.io/rss2026/pdfs/8_Any_ttach_Quick_End_effector.pdf).
  - ForceMimic, Bilaloglu and the knife paper all use rigid tool mounts.
  - A rigid peeler mount on the Franka flange (or a locked adapter in the Franka Hand) is the low-risk choice. RDP and FoAR show that grasped peelers work with tactile/force feedback, but they need grasp-pose variation handling.
- ForceMimic's HybridIL relies on Flexiv RDK hybrid force-position primitives. On an FR3 these would have to be re-implemented, for example as a libfranka torque controller with a force-controlled normal axis (inference).
- FR3 external-wrench estimates are likely noisier than a wrist F/T sensor. RDP notes its estimated wrench "has relatively larger noise compared to the force sensor mounted on the robot end effector" on Flexiv - [arXiv](https://arxiv.org/html/2503.02881v3). Inference: budget for a wrist F/T sensor (ATI Mini45, Bota) or tactile fingertips.
- Holding the cucumber:
  - With a single Franka, a fixture or clamp (MORPHeus jaw board, Bilaloglu fixture, OmniVTA "fixed" object) is the realistic option.
  - A human hand holding it (RDP, FORCA) adds perturbations that are hard to reproduce in sim.

### Gaps
- No paper reports a learned policy that peels cucumbers on a Franka with force or tactile input. That combination appears to be open.
- FR3-specific force estimation quality during peeling is not reported anywhere found.

## Q3. Do any peeling papers use simulation or sim-to-real, and how do they model the peel/skin?

### Takeaway
No paper found simulates vegetable peeling with a peeler or knife, and none transfers a peeling policy from sim to real.
- Simulation appears only around peeling:
  - Allegro reorientation in Isaac Gym (Vegetable Peeling).
  - A geometric orientation-field study on deformed point clouds (Bilaloglu).
  - Orange-peel fracture in AnisoMPM with no blade or robot.
- The closest templates for a Franka cucumber-peeling sim are knife-cutting simulators calibrated on real force data:
  - DiSECt: FEM with damage springs, real Panda cucumber.
  - MLS-MPM cutting with residual-RL zero-shot transfer: chicken deboning, RoboNinja.
  - SliceIt!: a dual simulator.

### Cited Findings
- The knife-peeling paper states that sim approaches for cutting "remain difficult to scale beyond simple insertion or cutting tasks due to the complexity of modeling cutting dynamics in deformable, heterogeneous produce" - [arXiv 2603.03280](https://arxiv.org/html/2603.03280v1).
- FORCA: no simulation - [arXiv](https://arxiv.org/html/2610.31841).

**Vegetable Peeling (Tao Chen)**
- Only the Allegro reorientation controller was trained in Isaac Gym, as a teacher-student RL setup - [arXiv 2407.07884](https://arxiv.org/abs/2407.07884).
- Sim setup:
  - 23 Objaverse vegetable meshes, 80-960 g.
  - No skin model; peeling was not simulated.
  - System identification of joint friction and damping; domain randomization.
- Peeling itself was teleoperated or scripted on the real robot.

**Latent Space RL for Inverse Material Estimation in Food Fracture Simulation**
- **Who:** Ramlal, Chen, Zelek (U Waterloo).
- **Venue:** arXiv 2606.16870, venue unverified - [arXiv](https://arxiv.org/abs/2606.16870).
- **Simulator:** AnisoMPM continuum damage, orange skin plus flesh, 9 material parameters. Actuation is "a gripper pulls the skin away" along a fixed arc. No blade, no robot.
- **Cost:** about 50,000 particles and about 156 s per run.
- **Results:** PPO recovery 0.642; with CMA-ES warm start 0.828.
- **Sim-to-real:** none.

**DiSECt**
- **Venue:** RSS 2021 - [arXiv 2105.12244](https://arxiv.org/abs/2105.12244); extended version arXiv 2203.10263, Autonomous Robots 2023 per citation - [arXiv](https://arxiv.org/abs/2203.10263).
- **Model:**
  - Neo-Hookean FEM.
  - Virtual-node duplication along a predefined cut surface, with springs that weaken to zero to form the crack.
  - The knife is an SDF.
- **Real robot:** Franka Panda with a knife, force from joint torques.
- **Calibration, knife-force NMAE before -> after:** cucumber 0.231 -> 0.050, apple 0.606 -> 0.083, banana 0.487 -> 0.160.
- **Optimized slicing:** reduced mean cucumber knife force by 46.7%.
- **Code:** [github.com/NVlabs/DiSECt](https://github.com/NVlabs/DiSECt).

**RoboNinja**
- **Venue:** RSS 2023 per citation - [arXiv 2302.11553](https://arxiv.org/abs/2302.11553).
- **Simulator:** differentiable MLS-MPM in Taichi. Flesh is elastoplastic (von Mises); the core, knife and board are SDFs.
- **Project:** [roboninja.cs.columbia.edu](https://roboninja.cs.columbia.edu/).

**SliceIt!**
- **Venue:** ICRA 2024 per citation - [arXiv 2404.02569](https://arxiv.org/abs/2404.02569).
- **Setup:** DiSECt cutting sim coupled to Gazebo over ROS, calibrated on real tomato, cucumber and potato force data.
- **Robot:** dual UR5e.
- **Training:** RL takes about 40 h.

**Chicken deboning**
- **Venue:** arXiv 2510.15376, "Accepted by ICRA 2026" per arXiv comment - [arXiv](https://arxiv.org/abs/2510.15376).
- **Method:** MLS-MPM plus CPIC multi-material cutting sim; residual PPO with discretized force observations and domain randomization.
- **Transfer:** zero-shot on a UR5 with a Robotiq FT 300.
- **Real-chicken success:** 0.10 nominal vs 0.50 adaptive.

**DexNinja**
- **Venue:** arXiv 2610.17208, venue unverified - [arXiv](https://arxiv.org/abs/2610.17208).
- **Method:** real2sim2real dexterous cutting with a differentiable MPM-based sim.
- **Results:** above 60% success; sim augmentation adds more than 20%.

**Other related simulators**
- **Food-cutting MPM simulators (no peeling):** CulinaryCut-VLAP (arXiv 2601.06451), MG-CD-MPM with phase-field damage and tool force/torque coupling (arXiv 2610.39975), STC-MPM with process-zone damage and no pre-inserted cut surface (arXiv 2610.06198) - [arXiv 2601.06451](https://arxiv.org/abs/2601.06451), [arXiv 2610.39975](https://arxiv.org/abs/2610.39975), [arXiv 2610.06198](https://arxiv.org/abs/2610.06198).
- **Peeling analogues (not produce):**
  - AutoPeel: wound-dressing removal with differentiable PBD and adhesion fracture, plus MPC - [arXiv 2409.14282](https://arxiv.org/abs/2409.14282).
  - Velcro peeling with tactile recurrent Q-learning (ICRA 2021 per citation) - [arXiv 2011.08319](https://arxiv.org/abs/2011.08319).
  - Velcro peel-front state estimation from force only - [arXiv 2506.05812](https://arxiv.org/abs/2506.05812).
- **Benchmarks:**
  - PhysLadder (Isaac Sim, dual-arm ARX R5) has no peeling. Its wiping task updates the visible surface state from contact and sliding - [arXiv 2610.34527](https://arxiv.org/abs/2610.34527).
  - ContactBench 1.0 lists deformables as future work - [arXiv 2610.24512](https://arxiv.org/abs/2610.24512).

### Inferences
- A Franka cucumber-peeling sim-to-real project would be novel. The literature offers three routes:
  1. **Physics route.** A two-region MPM with continuum damage (thin skin over flesh, as in the AnisoMPM orange), plus a blade SDF and a tool-force readout (as in the chicken-deboning MLS-MPM+CPIC). Calibrate against Franka joint-torque or F/T force profiles, as DiSECt did on a real Panda cutting a cucumber. A roughly 1 mm skin needs a fine grid, so RL in the loop probably needs a surrogate or a low-fidelity sim.
  2. **FEM route.** A peeler pass has a roughly known offset surface at peel depth. That fits DiSECt's limitation that the cut surface must be predefined (inference).
  3. **Abstracted route.** A rigid or FEM cucumber in MuJoCo/Isaac with a per-vertex "peeled" flag that flips when blade contact force and pose stay within a band, giving a coverage reward. Precedents: PhysLadder's wiping state update, MORPHeus coverage perception. Train a residual force policy with domain randomization (chicken-deboning recipe) around a vision-planned nominal stroke (Tao Chen PCA/spline).
- Calibration targets from real data:
  - ForceMimic human peeling forces around 10 N, HybridIL around 9 N average - [arXiv](https://arxiv.org/html/2410.07554).
  - FORCA's target band of 4.5-7.5 N with a 30 N limit - [arXiv](https://arxiv.org/html/2610.31841).

### Gaps
- No published constitutive parameters for cucumber skin vs flesh in a robotics simulator were found.
- Code was not confirmed for the orange-MPM, chicken-deboning or DexNinja simulators.
- SliceIt! real-robot success numbers were not retrieved.

## Q4. Older or classical peeling works a review committee should know

### Takeaway
Classical robotic peeling with a peeler comes mainly from Japanese labs:
- JSK humanoid (2013): sweet potato, peeler, rolling the vegetable on a board.
- UEC Kudoh lab (2019, 2021): dual-arm cooking robot, point-cloud pose estimation, rotation between strokes.

Purdue's Sawyer cucumber-peeling LfD work (2019, 2021) is the first learning-based cucumber peeling found. The University of Tokyo banana peeling (T-RO 2024) is the first long-horizon learned bimanual peel. No classical knife-based apple or potato peeling robot paper was found, and no KIT, Jozef Stefan, TUM or KAIST bimanual peeling paper was found.

### Cited Findings
- **Watanabe, Nagahama, Yamazaki, Okada, Inaba**, "Cooking Behavior with Handling General Cooking Tools based on a System Integration for a Life-sized Humanoid Robot", Paladyn J. Behavioral Robotics 4(2):63-72, 2013 (Univ. Tokyo JSK) - [De Gruyter](https://www.degruyterbrill.com/document/doi/10.2478/pjbr-2013-0013/html).
  - Dual 7-DoF arms with wrist force sensors.
  - The left hand holds the sweet potato on a board and the right hand peels.
  - The vegetable is rotated by rolling it on the board.
  - A peeled area is detected by image subtraction.
  - No peeling success numbers.
- **Dong, Yu, Takizawa, Kudoh, Suehiro**, "Food Peeling Method for Dual-Arm Cooking Robot", IEEE/SICE SII 2021, DOI 10.1109/IEEECONF49454.2021.9382700 - [UEC lab page](https://www.robo.lab.uec.ac.jp/en/publication/2021/sii/dong).
  - A "one peeling motion" along a pre-generated trajectory, plus a "rotation motion" that estimates food pose.
  - 5 food types.
  - Point cloud added for 3D, per MORPHeus - [arXiv](https://arxiv.org/html/2404.06570).
  - ForceMimic says this approach "depends heavily on preset assumption" - [arXiv](https://arxiv.org/html/2410.07554).
  - Predecessor: Yu et al., SI2019 (Japanese domestic) - [UEC](https://www.robo.lab.uec.ac.jp/en/publication/2019/si/yu/).
- **Balakuntala et al. (Purdue)**, one-shot cucumber peeling (2019, arXiv) and ICRA 2021 multimodal LfD (Sawyer, 100%) - see A13 and A14 - [arXiv 2103.01296](https://arxiv.org/html/2103.01296v2), [arXiv 1904.01846](https://arxiv.org/html/1904.01846).
- **Kim, Ohmura, Kuniyoshi**, banana peeling, T-RO 2024 per citation - see B3 - [arXiv](https://arxiv.org/html/2203.09749).
- **Straizys, Burke, Ramamoorthy (Edinburgh)**, "Surfing on an uncertain edge", ICRA 2020 - [arXiv 1909.07247](https://arxiv.org/html/1909.07247).
  - PR2 grapefruit segment scooping with a knife (cutting pulp from peel, not skin peeling).
  - DMP plus joint-effort classification.
  - Closed-loop success 36/50 (72%).
  - MORPHeus cites it under peeling - [arXiv](https://arxiv.org/html/2404.06570).
- **Wan et al.**, "Underactuated Humanoid Peeling Approach for Pickled Mustard Tuber Based on Metamorphic Constraints", Biomimetics 8(8):566, 2023 - [MDPI](https://www.mdpi.com/2313-7673/8/8/566).
  - A dedicated insert-clamp-tear mechanism, not an arm.
  - Results from ADAMS simulation only.
- **Industrial context:** tomato peeling methods (steam, lye, alternatives) are reviewed in Rock et al., Food Eng. Rev. 4:1-15, 2012, DOI 10.1007/s12393-011-9047-3. This is cited in the Biomimetics paper and was not read. Purpose-built peeling machines (taro, water chestnut) also exist in food engineering - [Sci Rep taro peeler](https://doi.org/10.1038/s41598-025-94100-3), [Sci Rep water chestnut](https://doi.org/10.1038/s41598-023-28472-9).
- **da Vinci grape skin peeling:** a demo video, not a paper - cited by MORPHeus - [arXiv](https://arxiv.org/html/2404.06570).

### Inferences
- The classical pipeline maps directly onto a Franka baseline, and the 2026 knife paper found that a heuristic planner alone reaches only 20%. The pipeline:
  - hold the produce with a second arm or a fixture
  - plan a stroke along the long axis from a point cloud
  - run constant-force or impedance contact
  - rotate and repeat
- That 20% figure is a useful honest lower bound to beat - [arXiv](https://arxiv.org/html/2603.03280v1).

### Gaps
- No full text was found for Dong 2021, so the robot model, food names and success numbers are unknown.
- No classical knife apple-peeling robot paper was found.
- No European bimanual (DMP/impedance) peeling paper was found despite targeted searches.

## Q5. What does each paper say fails?

### Takeaway
Failure modes reported across papers:
- losing contact or not engaging the blade (gaps, "peeling hesitation")
- excessive force (gouging, sensor damage, slip of the holding grip)
- breaking the strip when the controller switches modes or action chunks run open-loop
- curved, tapered ends and grooved surfaces
- tool slip or rotation in a parallel gripper

Force/tactile closed-loop control and rigid tool mounts are the main remedies reported.

### Cited Findings
- **RDP** - [arXiv](https://arxiv.org/html/2503.02881v3):
  - Vision-only DP predicts inaccurate trajectories with large contact forces.
  - DP with tactile input can "get stuck" switching contact states.
  - Open-loop chunks after a perturbation cause "broken cucumber peels".
  - Low-quality teleop data causes unstable force and "breaking halfway".
  - Case study: near the cucumber's end, a downward correction is needed to follow the surface.
- **ForceMimic** - [arXiv](https://arxiv.org/html/2410.07554):
  - DP baseline: excessive force (about 20 N average, above 40 N peaks) damaged the zucchini; one rollout broke its bottom. Some rollouts made no contact.
  - Peel breakage came from discontinuous output poses.
  - HybridIL ended force-position parameters prematurely, switching back to position control and breaking the peel.
  - Force-as-input policies failed because of a train/deploy force distribution mismatch.
- **FoAR:** baselines produce partial peels or failures without force feedback, and RISE gives "inaccurate position control" - [arXiv](https://arxiv.org/html/2411.15753v2).
- **TouchGuide** - [RSS PDF](https://www.roboticsproceedings.org/rss22/p078.pdf):
  - Peeling-arm force too high, so the holding arm's grip slips.
  - "Peeling hesitation".
  - "Peeling gap" on uneven cucumbers.
  - RDP fails from a bad initial blade angle.
- **ILA:** missed strip acquisition, strip breakage, lifting too early, tool overrun - [arXiv](https://arxiv.org/html/2610.18855v2).
- **TAO-Force:** the peeler misses the surface, or removes only a short peel - [arXiv](https://arxiv.org/html/2609.18497).
- **FORCA:** the vision-only agent loses contact (40.6%) and presses too hard, leaving gaps - [arXiv](https://arxiv.org/html/2610.31841).
- **Any-ttach:** the tool repeatedly slides on the surface without progress. With grasp-based holding, the tool slips or rotates in the gripper - [PDF](https://dex-manipulation.github.io/rss2026/pdfs/8_Any_ttach_Quick_End_effector.pdf).
- **Knife paper** - [arXiv](https://arxiv.org/html/2603.03280v1):
  - The skin-flesh boundary demands "precise force modulation and stable tool-object contact".
  - Peels are too thick or too thin.
  - Colour RGB hurts generalization; grayscale helps.
  - Kinesthetic demos fail on replay because compliance parameters are ambiguous.
  - A heuristic planner cannot run autonomously due to geometry variation.
- **MORPHeus** - [arXiv](https://arxiv.org/html/2404.06570):
  - Acorn squash grooves unreachable.
  - Items larger than the jaw slip out; small or spherical items evade the jaw.
  - Thin skin strips and carrot skin/flesh colour similarity confuse perception.
  - Eggplant peels do not fully detach, and eggplant leaves block the peeler.
  - Low stiffness fails on most foods.
- **Bilaloglu:** avocado skin detached midway; fixture compliance; the tool tip must be predefined - [Calinon PDF](https://calinon.ch/papers/Bilaloglu-ScienceRobotics2026.pdf).
- **Vegetable Peeling:** peeling applies "substantial pulling forces" on the held object - [arXiv](https://arxiv.org/html/2407.07884v1).
- **CAAI hand-arm:** without stiffness control, the peel on a tail-curved cucumber was only 3-4 cm - [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70185).
- **Kim banana:** missed reaches and broken necks; necks had to be pre-cut - [arXiv](https://arxiv.org/html/2203.09749).
- **Sensorless bilateral:** without force, the policy stalls where the gripper and peeler overlap visually. Some runs jump to the "finished" state before touching the cucumber - [arXiv](https://arxiv.org/html/2507.06174).

### Inferences
- The recurring physical issues are depth/contact regulation on a curved, tapering, slippery surface, and holding the produce against pulling forces. For a Franka sim-to-real effort they point to:
  - a normal-axis force or admittance loop at or above 100 Hz
  - a rigid peeler mount
  - a fixture or clamp for the cucumber
  - a 6-15 N force band, from the ForceMimic and FORCA numbers above
- The tapered-end failures (RDP case study, CAAI tail-curved cucumber) suggest that sim training should randomize cucumber curvature and taper, not just pose (inference).

### Gaps
- No paper reports peel thickness in mm for peeler-based cucumber peeling. Only length or coverage is reported. The knife paper uses a "perceived thickness" proxy.
- Cucumber surface friction (wet or waxy skin) is not quantified as a failure source in any paper found.
