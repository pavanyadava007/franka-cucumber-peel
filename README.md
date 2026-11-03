# Franka cucumber peeling: GELLO + RDP, simulation to real

Two Franka Emika Panda arms (7 joints each) peel a cucumber. The left arm holds and rolls the cucumber, the right arm
peels with a mounted peeler and a wrist force sensor. Demonstrations go through the GELLO joint interface, the policy
is Reactive Diffusion Policy (RDP, RSS 2025), and the same code paths are prepared for two real Pandas.

Status (honest): **everything below runs in simulation on this PC. Nothing has run on a real robot yet.** The demos
are from a scripted expert driven through the GELLO path, not from people. The skin is a rule-based model, not a
simulated material. See docs/SIMULATION.md and docs/DATA.md.

## What is in here

| Part | Where | State |
|---|---|---|
| Bimanual Panda cell in MuJoCo, 7 cucumber shape classes, skin model, force sensing | `peel/` | working, 12 sim tests |
| Scripted expert + orchestrator (makes the demos) | `peel/expert.py` | working on all shape classes |
| GELLO path: real GELLO arms drive the sim; replay check | `scripts/teleop_gello_sim.py` | replay verified; real Dynamixel reading untested here |
| Demo collection and RDP datasets (RDP's zarr layout) | `scripts/collect_demos.py`, `scripts/make_rdp_dataset.py` | working |
| RDP training (official code, unmodified) + Diffusion Policy baseline | `scripts/train_rdp.sh`, `rdp_configs/` | working on a 4 GB GTX 1650 |
| Policy evaluation in sim, stratified by shape | `scripts/eval_policies.py`, `scripts/make_results.py` | see docs/RESULTS.md |
| Real two-Panda layer (franky backend, dry run by default, safety limits) | `peel/real_robot.py`, `scripts/run_real.py` | mock-tested only |
| Public dataset audit (which "cucumber peeling" sets really are) | `docs/DATASET_AUDIT.md` | done |

## Setup

```bash
git clone --recurse-submodules https://github.com/pavanyadava007/franka-cucumber-peel.git && cd franka-cucumber-peel
uv venv -p 3.10 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python mujoco==3.17.0 numpy==1.26.4 scipy "zarr<3" numcodecs hydra-core einops dill \
  "diffusers<0.32" timm accelerate opencv-python-headless imageio imageio-ffmpeg av termcolor tensorboard tqdm numba h5py \
  transforms3d loguru pillow pyzmq tyro dynamixel-sdk matplotlib pandas pytest ruff threadpoolctl psutil click \
  "huggingface_hub<0.26" peft scikit-image wandb
bash scripts/fetch_models.sh                    # Franka Emika Panda from MuJoCo Menagerie (pinned)
git submodule update --init                     # official RDP (824c5e8) and GELLO (204f53a) code, pinned
export MUJOCO_GL=glfw                           # WSL2: CPU rendering; use egl on a normal Linux GPU box
```

## Run it end to end

```bash
python -m pytest -q                                              # sim + real-robot mock tests
python scripts/make_video.py --seed 0                            # watch the scripted expert peel
python scripts/collect_demos.py --n 84 --start-seed 2002 --workers 5
python scripts/make_rdp_dataset.py --kind stroke                 # data/rdp/peel_stroke_zarr
bash scripts/train_rdp.sh all                                    # RDP stage 1 + 2, then the DP baseline
python scripts/eval_policies.py --method expert_force
python scripts/eval_policies.py --method expert_noforce
python scripts/eval_policies.py --method rdp --ckpt runs/rdp_ldp/checkpoints/latest.ckpt --video-first
python scripts/eval_policies.py --method dp  --ckpt runs/dp/checkpoints/latest.ckpt --video-first
python scripts/make_results.py                                   # writes docs/RESULTS.md
```

With real GELLO arms: `python scripts/teleop_gello_sim.py --left-port ... --right-port ... --record`.
On the real robots: read docs/REAL_ROBOT.md first, then `python scripts/run_real.py status`.

## Trained models and data

Git holds the code, docs, results and videos. The trained RDP checkpoints and the raw demo episodes are too
large for git and are attached to the GitHub Release of this repository (`gh release download -R
pavanyadava007/franka-cucumber-peel`). Unpack them into `runs/` and `data/`.

## Documents

- docs/ARCHITECTURE.md: orchestrator, stroke skill, controller; how RDP is used
- docs/SIMULATION.md: the cell, controller, cucumber shapes, the skin model and its assumptions
- docs/DATA.md: exactly how the dataset is created
- docs/DATASET_AUDIT.md: public datasets checked frame by frame
- docs/RESULTS.md: evaluation results (generated)
- docs/REAL_ROBOT.md: sim-to-real guide for two Pandas (hardware, calibration, GELLO, safety, protocol)
- docs/research/: the literature review (top 10 papers, committee ranking) and its research notes

## Credits and licences

Franka Emika Panda model: MuJoCo Menagerie (Apache-2.0). RDP: Xue et al., RSS 2025, MIT licence. GELLO: Wu et al.,
MIT licence. This project's own code has no licence file yet; add one before publishing.
