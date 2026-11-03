# Public robot cucumber peeling datasets: audit

Audit date: 2026-11-03. Auditor: Claude (agent), for the franka-cucumber-peel project.

Method. For every dataset I read the metadata myself (info.json, tasks, episode tables, READMEs). I also downloaded a small slice (one parquet, one video file, or a few files read by HTTP range requests) and looked at real image frames. No verdict below rests on a dataset name. Every number comes from files I opened. Where I could not check something, I say so.

Safety. Each download went into its own folder under `data/external/<short_name>/`. I never executed anything from a download. No pickle files were loaded (the RDP raw subset and the How to Peel data are pickles). Python ran with `python -I` from `tools/audit/`.

Contact sheets (labelled frames I looked at) are in `media/dataset_audit/<short_name>.jpg`.

## Summary table

| # | Dataset id | Robot (stated / seen) | Episodes | fps | Cameras | Force / tactile channels | Licence | Task string(s), verbatim | VERDICT | Evidence (frames viewed) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `WendiChen/reactive_diffusion_policy_dataset` (RDP, file `dataset_mini/peel_v3_downsample2_zarr.zip`) | README: 2x Flexiv Rizon 4 with GRAV gripper. Seen: two grey arms with yellow-green rings; only one arm moves | 2 in the mini zarr (388 frames). Full set: not counted | 12 (README: recorded at 24, downsample 2). Stored timestamps are unusable (see notes) | `external_img`, `left_wrist_img` (240x320 RGB) | `left/right_robot_tcp_wrench` (6), `left/right_robot_gripper_force` (1), tactile images `left_gripper1_img` (9x7 markers) and `left_gripper2_img` (5x5 markers), marker arrays (63x2, 25x2), PCA embeddings (15 each) | MIT | none stored in the zarr | **VERIFIED cucumber peeling** | ep0 frames 0, 50, 90 to 201 and ep1 frames 202 to 387 (wrist + external). Robot grasps a white loop peeler from a holder. A human hand in a white glove holds the cucumber against the blade. A dark skin strip trails from the cucumber and light peeled bands are visible. |
| 2 | `qiaojunyu/ForceVLA-real-data`, folder `data_lerobot/flexiv_peel_cucumber_inputForce` | `flexiv` / single grey arm with light rings, black parallel gripper | 43 (25,869 frames) | 30 | `observation.image` (third person), `observation.wrist_image` (480x640 PNG in parquet) | `observation.state[7:13]` = `f_ext_base_frame` (6) | Apache-2.0 | `Peel cucumber skin.` | **VERIFIED cucumber peeling** | ep19 (frames sampled across all 493, about 20 viewed on both cameras). Robot picks a blue Y-peeler from a cup and strokes along a cucumber lying on a blue board. A metal block holds one end. Lighter peeled bands and dark peel flakes are visible. I could not isolate fresh skin removal inside ep19 alone. |
| 3 | `henryyiw/task_cucumber_peel_lerobot` (+ `_new` variants) | `ur5` / white arm, only partly visible; make not confirmed from frames | 120 (56,689 frames) | 30 | `realsense_0`, `realsense_1`, `realsense_2` (stored 240x320 PNG) | `gelsight` image (240x320); no wrench | none stated (no README) | `peel the cucumber into strips` | **VERIFIED cucumber peeling** | ep8 frames 0, 90, 230 to 362 and ep115 frames 0 to 327. Cucumber on a motorised spindle (motor box at one end, metal bracket at the other). Robot takes a wooden peeler from an orange holder. A thin strip lifts behind the blade (ep8 f255, f280); a light peeled band at f362. Raw `_new` repo, 640x480 frames: strip curling off a green-handled peeler. |
| 4 | `JensenYuan/peel_tcp_in_state_lerobot` | `task_peel` (no robot named) / same rig as #3 | 61 (10,952 frames) | 30 | `realsense_0/1/2` (240x320 PNG) | `gelsight` image; no wrench | none stated (no README) | `peel the cucumber into strips` | **VERIFIED cucumber peeling** | ep43 frames 0 to 124. Robot holds a green-handled swivel peeler and strokes a spindle-mounted, mostly peeled cucumber. A pile of skin ribbons lies on the table. A small piece flies off at f85. |
| 5 | `siyitian/peel_cucumber` | `r5_single_arm` / small black single arm on a pink grid table | 51 (29,968 frames) | 30 | `top`, `wrist_right` (240x320 AV1) | `observation.force.right.df` (204 floats, undocumented, about 5 % non-zero) | Apache-2.0 | `Peel the cucumber.` | **not peeling** | ep0 (t 0 to 19.2 s), ep25 (t 0 to 23.8 s), ep50 (t 0, 8, 15.5 s). Robot picks a ridged potato chip from a small stand and puts it on a striped plate. No cucumber and no peeler in any frame. The task string is wrong. |
| 6 | `VibeCuisine/cucumber-peel-DAgger-iter2-v2-trim` | `vibeboard_follower_tilt` / servo arm (SO-100 style joint names + tilt) on a gantry board | 188 (58,994 frames) | 20 | `base`, `top`, `wrist` (480x640 AV1) | none (aux: `elevator_mm`, `spinner_active`, `limit_home`, `limit_gripper`) | none stated | `peel the skin of the cucumber with several strokes, starting close to the blue gripper and moving vertically along the cucumber toward the robot origin` | **VERIFIED cucumber peeling** | ep170 and ep185 (top + wrist, start to end). Cucumber lies on a wooden board. A blue clamp on a rail holds its far end. The end effector carries a metal peeler blade on a toothed tilt gear (fingers open). Skin strips come off and pile up. |
| 7 | `tompollok/vibecuisine-vibeboard-042226-peel-cucumber-dual-gripper` | `so_follower` / same VibeCuisine rig as #6 | 36 (17,036 frames) | 20 | `top`, `wrist`, `base` (480x640 H.264) | none | Apache-2.0 | `Peel the cucumber with several strokes, starting close to the blue gripper and moving vertically along the cucumber toward the robot origin` | **VERIFIED cucumber peeling** | ep0, ep18, ep35 (start, 33 %, 66 %, end on all 3 cameras). ep0 starts dark green and ends with peeled bands. The base view shows a skin strip curling off the blade. ep35 still looks unpeeled at its last frame (not checked why). |
| 8 | `BAAI-DataCube/AgiBotWorld-Beta_G1_task_550_Peel_fruits_and_vegetables` | `a2d`, README "G1", gripper / dual 7-DoF arms, white arm + black grippers | 863 (1,391,773 frames). Cucumber subset: 374 by annotation | 30 | `head`, `hand_left`, `hand_right` (480x640) + 5 fisheye (768x960), all AV1 | none as wrench; `observation.states.joint.current_value` (14) | Card: no licence field; gating prompt says research and academic use only, no commercial use. Not gated at audit time | `Peel fruits and vegetables \| The robot faces the cutting board, with fruits and vegetables on the board, a peeler in the storage basket, and the storage basket on the countertop.` | **VERIFIED cucumber peeling** (cucumber subset only; the rest is radish, potato, carrot) | ep861 head t 1 to 55 s: left gripper pins a cucumber on the board, right arm takes a Y-peeler and strokes; at the end a skin strip lies on the board and a pale band shows. ep0 hand_left t 30 s: a strip hangs off a cucumber (different kitchen). ep862: carrot, matching its annotation. |
| 9 | `Jefferzn/peel` | `yam_bimanual` / two YAM arms | 1 (599 frames) | 10 | `cam_0`, `cam_1` (480x640 AV1) | none | Apache-2.0 | `peel the cucumber` | **VERIFIED cucumber peeling** | t 0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 59.5 s on cam_0, plus cam_1. One arm holds the cucumber upright in the air. The other holds a red Y-peeler with a blue handle. At t 0 the cucumber is uniformly dark. At t 59.5 s it has pale peeled patches and a strip at the tip. |
| 10 | How to Peel with a Knife (Berkeley, `ToruOwO/how-to-peel`, Dropbox `peel-data`) | README: Kinova Gen3 7-DoF, ATI mini45 F/T, 2x RealSense D405 wrist | not checked | policy 10 Hz, cameras 30 fps (from repo constants) | 2 wrist RGB-D | ATI mini45 6-axis F/T (stated) | no licence file in the repo | none read (data not opened) | **could not verify** | Not downloaded. Cucumber data is 20 GB + 9.25 GB in tar.zst parts. No small sample in the folder. Uses a knife, not a peeler, per the README. |
| 11a | FoAR sample data (`Alan-Heoooh/FoAR`, Google Drive `[data]FoAR/peel/train.zip`) | not stated in README / black linkage gripper, arm not identifiable | 3 in the sample | color about 4 to 5 Hz by frame count (218 frames over about 47.5 s); F/T at 200 Hz | 2 RGB-D cams (1280x720 PNG) | `force_torque_tcp_joint_timestamp.npy`: 34 columns at 200 Hz, first 6 look like F/T | no licence file in the repo | none (RH20T style folders; `metadata.json` holds `rating` and `finish_time` only) | **VERIFIED cucumber peeling** | scene_0002 both cameras start, 40 %, 70 %, end; scene_0004 two frames. Robot picks a blue peeler from a white stand. The cucumber lies on a blue board between two yellow blocks. Skin is intact at the start and pale peeled patches show at the end. |
| 11b | ForceMimic processed data (`ForceMimic/hybridil`, Google Drive `ForceMimic_example_data/zucchini_shave.hdf5`) | README: Flexiv RDK for evaluation; data from the ForceCapture force-motion capture system / no arm or hand visible in the cropped point cloud | 139 demos (`demo_XX_shave_YY`) | not stored in the parts I read | 1 L515 depth camera, stored as coloured point clouds (10,000 points per frame), no RGB images | `pyft_fs` (3) + `pyft_ts` (3) force and torque on the tool | code MIT; data licence not stated | none (group names say `shave`) | **peeling but not cucumber** | demo_03_shave_00 frames 0, 90, 180, 261 rendered from the point cloud (range read, file not downloaded). Blue Y-shaped tool on a yellow-green zucchini; a pale peeled band at the end. Vegetable identity rests mainly on the file name. |

## Observed RDP zarr schema (the main build mirrors this)

Source: `dataset_mini/peel_v3_downsample2_zarr.zip`, 202,527,917 bytes zipped, 0.27 GB unpacked. The zip holds one directory store `peel_v3_downsample2_zarr/replay_buffer.zarr` (zarr format 2). There are no group or array attributes.

```
replay_buffer.zarr/
  .zgroup
  data/                                     (group)
    action                         (388, 4)            float32  chunks (10000, 4)          blosc zstd clevel 3 shuffle
    external_img                   (388, 240, 320, 3)  uint8    chunks (100, 240, 320, 3)  blosc zstd clevel 3 shuffle
    left_gripper1_img              (388, 240, 320, 3)  uint8    chunks (100, 240, 320, 3)  blosc lz4  clevel 5 shuffle
    left_gripper1_initial_marker   (388, 63, 2)        float32  chunks (10000, 63, 2)      blosc lz4  clevel 5 shuffle
    left_gripper1_marker_offset    (388, 63, 2)        float32  chunks (10000, 63, 2)      blosc lz4  clevel 5 shuffle
    left_gripper1_marker_offset_emb (388, 15)          float32  chunks (10000, 15)         blosc lz4  clevel 5 shuffle
    left_gripper2_img              (388, 240, 320, 3)  uint8    chunks (100, 240, 320, 3)  blosc lz4  clevel 5 shuffle
    left_gripper2_initial_marker   (388, 25, 2)        float32  chunks (10000, 63, 2)      blosc lz4  clevel 5 shuffle
    left_gripper2_marker_offset    (388, 25, 2)        float32  chunks (10000, 63, 2)      blosc lz4  clevel 5 shuffle
    left_gripper2_marker_offset_emb (388, 15)          float32  chunks (10000, 15)         blosc lz4  clevel 5 shuffle
    left_robot_gripper_force       (388, 1)            float32  chunks (10000, 1)          blosc zstd clevel 3 shuffle
    left_robot_gripper_width       (388, 1)            float32  chunks (10000, 1)          blosc zstd clevel 3 shuffle
    left_robot_tcp_pose            (388, 9)            float32  chunks (10000, 9)          blosc zstd clevel 3 shuffle
    left_robot_tcp_vel             (388, 6)            float32  chunks (10000, 6)          blosc zstd clevel 3 shuffle
    left_robot_tcp_wrench          (388, 6)            float32  chunks (10000, 6)          blosc zstd clevel 3 shuffle
    left_wrist_img                 (388, 240, 320, 3)  uint8    chunks (100, 240, 320, 3)  blosc lz4  clevel 5 shuffle
    right_robot_gripper_force      (388, 1)            float32  chunks (10000, 1)          blosc zstd clevel 3 shuffle
    right_robot_gripper_width      (388, 1)            float32  chunks (10000, 1)          blosc zstd clevel 3 shuffle
    right_robot_tcp_pose           (388, 9)            float32  chunks (10000, 9)          blosc zstd clevel 3 shuffle
    right_robot_tcp_vel            (388, 6)            float32  chunks (10000, 6)          blosc zstd clevel 3 shuffle
    right_robot_tcp_wrench         (388, 6)            float32  chunks (10000, 6)          blosc zstd clevel 3 shuffle
    target                         (772, 4)            float32  chunks (10000, 4)          blosc zstd clevel 3 shuffle
    timestamp                      (388,)              float32  chunks (10000,)            blosc zstd clevel 3 shuffle
  meta/                                     (group)
    episode_ends                   (2,)                int64    chunks (10000,)            blosc zstd clevel 3 shuffle
```

What the arrays contain (from my checks on this file):

- `meta/episode_ends` = `[202, 388]`. Episode i covers rows `[episode_ends[i-1], episode_ends[i])`, so the lengths are 202 and 186. This is the diffusion_policy ReplayBuffer layout.
- `*_robot_tcp_pose` (9) = xyz in metres + 6D rotation (two rows of the rotation matrix). The 6D layout matches `use_6d_rotation: True` in the RDP task config. The left pose x ranges 0.50 to 0.63 m, y -0.23 to 0.22 m, z 0.14 to 0.41 m.
- `action` (4) = absolute left TCP xyz (m) + left gripper width (m). The RDP config `real_peel_image_dp_absolute_12fps.yaml` declares `action.shape: [4]`. In the data, action stays within about 1 mm of the measured left TCP xyz and width (mean absolute difference 0.82 mm at the same row, 1.25 mm one row later). Orientation is not in the action.
- `target` (4) has 772 rows, not 388. Its jump between episodes is at row 401, against row 201 for `action`. Its odd rows in episode 0 equal the downsampled left xyz + width exactly. So `target` seems to be stored at the raw 24 fps rate (402 + 370 rows) and is not row-aligned with the other arrays. The README listing for the full set shows `target` with the same length as the other arrays, so this may only affect the mini file. I did not check the full file.
- `timestamp` is float32 Unix time. At 1.7e9 the float32 step is 128 s, so every frame in an episode has the same value (two unique values in the file). fps cannot be derived from it. My estimate from position change over measured TCP speed is about 0.079 s per frame (about 12.7 Hz). This is consistent with the README (24 fps recording, downsample ratio 2, so 12 fps). **When mirroring, store timestamps as float64.**
- Only the left arm moves. All `right_robot_*` arrays are constant over both episodes (right TCP velocity under 0.003). The right gripper width stays at 0.1 m.
- `left_robot_tcp_wrench` (6) = force (N) and torque (Nm). Left |F| percentiles 50 / 90 / 99 / max: 3.28 / 5.62 / 11.14 / 11.87 N. Fz ranges -8.75 to 4.66 N. The frame (TCP or base) is not stated in the README. I did not check it.
- `left_robot_gripper_force` (1): median 24.46 N while the peeler is held, -1.21 to 30.17 N overall.
- `left_gripper1_*` is a 9x7 marker gel (63 markers, GelSight Mini per the RDP README). `left_gripper2_*` is a 5x5 marker gel (25 markers, McTac per the config name `..._one_gelsight_one_mctac_24fps`). `*_marker_offset_emb` (15) are PCA embeddings of the marker offsets (`n_components: 15` in the config). Note that the `left_gripper2_*marker*` chunk shape is (10000, 63, 2) while the data is (388, 25, 2).
- Image channel order: the frames look right when read as RGB (green cucumber, blue sleeve).

## Per-dataset details

### 1. RDP (`WendiChen/reactive_diffusion_policy_dataset`)

- Repo total is 198 GB. I took only `dataset_mini/peel_v3_downsample2_zarr.zip`. I did not open `dataset_mini/peel_v3.zip` (1.16 GB). The README says raw subsets are `seq_XX.pkl` pickles, and loading an untrusted pickle runs code. The zarr already gave clear evidence, so the raw file was not needed.
- Other peel files: `dataset_mini/peel_v3_downsample1_zarr.zip` (0.40 GB), `dataset_full/peel_v3_downsample2_zarr.zip` (6.81 GB), `dataset_full/peel_v3_downsample1_zarr.zip` (13.55 GB), `dataset_full/peel_v3.zip` (39.0 GB raw).
- Scene: the robot grasps a white 3D-printed peeler (loop blade) from a holder. It lifts it to a fixed height. A person in a white glove and blue sleeve brings the cucumber to the blade and moves it along. The robot does not hold the cucumber. Holding and pushing the cucumber is done by a human hand.
- Contact sheet: `media/dataset_audit/rdp.jpg`.

### 2. ForceVLA (`qiaojunyu/ForceVLA-real-data`, `flexiv_peel_cucumber_inputForce`)

- LeRobot v2.1, one parquet per episode with PNG images embedded as `struct{bytes, path}`. I downloaded `episode_000019.parquet` (338 MB, the smallest; 493 frames, 16.4 s). Total folder size is 17.6 GB.
- `observation.state` (13, float64) names: `current_eef_pose` (6), `gripper_width` (1), `f_ext_base_frame` (6). Split: [0:3] xyz in m, [3:6] three rotation values (representation not documented), [6] gripper width (0.014 to 0.0999 m in ep19), [7:13] external force and torque in the base frame.
- `action` (7, float64): `target_eef_pose` (6) + `target_gripper_width` (1).
- Force in ep19: force x -0.06 to 6.15, y -4.74 to 9.12, z -4.45 to 10.07 (assumed N). |F| sampled every 30 frames peaks at 9.48.
- A sibling folder `flexiv_peel_cucumber_noForce` also exists (same episode table size). I did not open it.
- Who holds the cucumber: a fixture (a metal block at one end, on a blue board). The robot holds the peeler. At the end of ep19 the robot puts the peeler down on the board.
- Contact sheet: `media/dataset_audit/forcevla.jpg`.

### 3. `henryyiw/task_cucumber_peel_lerobot` and variants

- LeRobot v3.0, one data file per episode (`file-NNN` = episode NNN), images embedded as PNG. Downloaded `file-115.parquet` (123 MB, ep115) and `file-008.parquet` (131 MB, ep8).
- No README, so the 16-dim `state` is undocumented. My reading, checked against the raw robot log column names in `task_cucumber_peel_new` (`j1_rad..j6_rad, tcp_x_m, tcp_y_m, tcp_z_m, tcp_rx_rad, tcp_ry_rad, tcp_rz_rad, gripper_mm`):
  - [0:6] six joint angles (rad); joint 6 is constant within an episode.
  - [6] gripper value in raw units (369 to 840 in ep115).
  - [7:10] TCP xyz. The values (276 to 497, -258 to 180, 194 to 441) look like millimetres.
  - [10:16] 6D rotation. I checked that [10:13] and [13:16] are unit vectors and orthogonal (mean |u.v| below 2e-8).
- `actions` (7) equals `state[:7]` exactly at the same row (mean difference 0 at lag 0). So "action" is the current joint state + gripper, not a future target. Shift it by one or more steps before training.
- Rig: the cucumber is mounted on a horizontal spindle between a black motor box and a metal bracket. It is probably rotated between strokes, but I did not verify this. Tools: in the 120-episode set a wooden-handled peeler from an orange holder. In the raw `_new` set a green-handled swivel peeler.
- Variants (all public, all task `peel the cucumber into strips`, robot_type `ur5`):
  - `task_cucumber_peel_lerobot_full_state`: 120 episodes, state 19.
  - `task_cucumber_peel_new`: raw data, 169,445 files, 67.55 GB. Folders `green/` (30 eps), `split_1/` (61), `split_2/` (65). Each episode has `meta.json` (cameras 640x480 at 30 fps, GelSight at 60 fps), `robot_log.csv`, `gripper_events.csv` and PNG folders. I downloaded 20 PNGs from `green/episode_000` and `split_1/episode_000`. The `split_1` episode shows an unpeeled cucumber and a skin strip curling off the blade. The `green` episode shows a nearly fully peeled cucumber.
  - `task_cucumber_peel_new_lerobot` (156 eps, 42,244 frames, state 7), `..._new_lerobot_with_tcp` (state 16), `..._new_lerobot_full_state` (state 19). I read their meta only.
- Contact sheet: `media/dataset_audit/henry.jpg`.

### 4. `JensenYuan/peel_tcp_in_state_lerobot`

- LeRobot v2.1, per-episode parquet with embedded PNG. Downloaded `episode_000043.parquet` (47 MB, 125 frames, 4.1 s).
- Same schema as #3 (state 16 with the same layout, actions 7 equal to `state[:7]` at the same row). The same rig is visible: spindle, metal bracket, white robot, green-handled peeler, plastic sheet on the table.
- I cannot tell from the files whether these episodes overlap with the henryyiw `_new` episodes.
- Contact sheet: `media/dataset_audit/jensen.jpg`.

### 5. `siyitian/peel_cucumber`

- The repo holds both a v2.1 style per-episode layout and a v3.0 single file layout. I used the per-episode videos for ep0, ep25 and ep50 (about 2 to 3.5 MB each) and the v3.0 data parquet (1.9 MB).
- All three episodes show a pick-and-place of a potato chip onto a plate. The task string `Peel the cucumber.` does not match the content.
- `observation.force.right.df` (204 floats) is about 5.4 % non-zero. Its layout is not documented. It may be a distributed force field (for example 68 points x 3 axes), but I did not verify this.
- Contact sheet: `media/dataset_audit/siyitian.jpg`.

### 6. `VibeCuisine/cucumber-peel-DAgger-iter2-v2-trim`

- LeRobot v3.0, AV1 videos concatenated per file. Downloaded `videos/observation.images.top/chunk-000/file-004.mp4` (72 MB, eps 166 to 187) and `videos/observation.images.wrist/chunk-000/file-003.mp4` (61 MB, eps 164 to 187).
- Built by "Vibe Data Studio" from 6 component datasets (117 episodes from rig `jetson1`, 71 from rig `vibepi`). The recipe is in `meta/vibedata/collection.json`. It is a DAgger iteration, so some episodes may include policy rollouts with corrections. The card does not say which.
- State and action (7) are joint positions: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper, tilt`. No force channel.
- Rig: a gantry board with a rail. A blue clamp on the rail holds the far end of the cucumber. The arm's end effector has a metal peeler blade on a toothed gear (the `tilt` joint). The gripper fingers stay open during strokes. The aux channels `elevator_mm` and `spinner_active` suggest actuated parts. I did not check what they move.
- Contact sheet: `media/dataset_audit/vibecuisine_dagger.jpg`.

### 7. `tompollok/vibecuisine-vibeboard-042226-peel-cucumber-dual-gripper`

- LeRobot v3.0, H.264. Downloaded all three video files (114 MB, all 36 episodes).
- Same rig as #6, with 6 joints (no tilt in the state).
- A sibling repo `...-dual-gripper2` exists. I did not open it.
- Contact sheet: `media/dataset_audit/tompollok.jpg`.

### 8. AgiBot World Beta task 550 (`BAAI-DataCube/...task_550_Peel_fruits_and_vegetables`)

- Access: the HF API reports `gated: False`, and the meta files and one video downloaded without login or accepting terms. So I did not have to get past any gate. The card still carries an `extra_gated_prompt`: "You agree to use this dataset for research and academic use only and not for any commercial use." Treat that as the intended terms. The card has no licence field. The upstream `agibot-world/AgiBotWorld-Beta` is gated (`auto`).
- Size 46.98 GB. I downloaded `videos/observation.images.head/chunk-000/file-034.mp4` (46 MB, ep861 cucumber + ep862 carrot). I also read three single frames from ep0 `hand_left` / `hand_right` by letting ffmpeg seek over HTTP (only the needed byte ranges were fetched).
- The episodes table has an `action_config` column with sub-task segments (text, skill, start and end frame). I parsed it into `data/external/agibot550/episode_vegetables.csv`. Counts: cucumber 374, green radish 189, white radish 155, potato 92, organic carrot 53. Every cucumber episode has the same six steps: left arm grabs the left side of the cucumber on the board, right arm takes the peeler, right arm peels, right arm "tidies" the attached skin, right arm puts the peeler back, left arm releases. The cucumber peel segment is 196 to 671 frames long (median 392.5).
- State: per-arm joint position (7 + 7), joint current (7 + 7), end-effector position (2x3) and quaternion xyzw (2x4), gripper position (2), head (2), waist (2), base pose. Actions: joint position (14), end-effector pose, gripper, head, waist, base velocity. No wrench sensor channel. Joint currents may work as a rough effort proxy (not checked).
- At least two scenes: a tiled kitchen with a blue board (ep861, 862) and a kitchen by a window with a wooden board (ep0).
- Contact sheet: `media/dataset_audit/agibot550.jpg`.

### 9. `Jefferzn/peel`

- LeRobot v3.0, 1 episode, 59.9 s, 10 fps. Downloaded both videos (6.4 MB).
- State and action (14): `Lleft_j0..j5, Lleft_gripper, Lright_j0..j5, Lright_gripper`.
- This is the only dataset here where the cucumber is held by a robot gripper in the air and peeled by the other arm. The video is blurry during motion. Peeling is clear only by comparing the first and last frames.
- Contact sheet: `media/dataset_audit/jefferzn.jpg`.

### 10. How to Peel with a Knife (Berkeley)

- From the README and the Dropbox folder page (not the data itself):
  - Hardware: Kinova Gen3 7-DoF, impedance control, ATI mini45 F/T sensor, two RealSense D405 wrist cameras, SpaceMouse teleoperation. The tool is a knife.
  - Dropbox folder `peel-data`: `apple.tar.part-aa/ab/ac` (20 + 20 + 18.23 GB), `cucumber.tar.part-aa/ab` (20 + 9.25 GB), `potato.tar.part-aa..af` (5 x 20 + 2.67 GB). The download script still labels potato "COMING SOON". The parts are a split zstd tar.
  - Format, from `episode_storage.py`: one folder per episode, `<output_dir>/<score>/<start_time>/`, with per-step `data_<i>.pkl` (pickle of `version, timestamp, observation, action`), `<cam>_rgb/` PNGs, `<cam>_depth/` NPY, preview MP4s, optional SAM mask MP4s, and `data_config.json`. Policy control is 10 Hz, arm control 1 kHz, camera capture 30 fps (`constants.py`).
- No small sample exists, so I saw no frames. The data is pickled, so it needs a sandbox and a trusted source before loading.
- No contact sheet.

### 11a. FoAR sample data

- The Drive folder `[data]FoAR` has `peel/` (`calib/`, `train.zip` 1.1 GB = 1,212,517,095 bytes) and `wipe/`. The file is over the 300 MB limit, so I did not download it. Drive answers HTTP range requests, so I read the zip's central directory and 14 single members (12.5 MB in total) with `tools/audit/http_zip.py`. The full member list is in `data/external/foar/train_zip_listing.txt`.
- Contents: 3 episodes (`task_0701_user_0099_scene_0002/0003/0004_cfg_0001`), 17,695 zip members. Each has two cameras (`cam_043322070878`, `cam_750612070851`) with `color/` (164 to 222 PNG per camera per episode, 1280x720), `depth/`, `tcp/`, `joint/`, `gripper_info/`, `gripper_command/` (3,850 files each in total across cameras and episodes), plus `high_freq_data/force_torque_tcp_joint_timestamp.npy`, `robot_command/`, `pedal_command/`, `metadata.json` (only `finish_time` and `rating`, both episodes I read have rating 9), and `timestamp.txt` (calibration timestamp).
- `force_torque_tcp_joint_timestamp.npy` for scene_0002: 13,115 x 34 float64. The last column is a millisecond timestamp with a median spacing of 5 ms (200 Hz) over 52.6 s. The first 6 columns look like force and torque (|F| median 1.88 N, 99th percentile 19.3 N, max 25.8 N). The next 7 look like TCP xyz + quaternion. I did not decode the remaining columns. The README only says "force/torque, tcp, joint, timestamp".
- Contact sheet: `media/dataset_audit/foar.jpg`.

### 11b. ForceMimic processed data

- The Drive folder `ForceMimic_example_data` has `cvt_robomimic.py` (a converter script, not run), `readme.md` (two lines: visualise with ForceCapture `visualize_merge.py`, convert with `cvt_robomimic.py` for HybridIL) and `zucchini_shave.hdf5` (9.0 GB = 9,653,511,124 bytes). The file is far over the 300 MB limit, so I did not download it. I read its structure and 4 frames of one demo by HTTP range requests (`tools/audit/http_h5.py`, about 6 MB in total).
- Structure: `/data` holds 139 groups named `demo_XX_shave_YY`. Each group I opened (`demo_03_shave_00`, 262 frames) has 11 uncompressed datasets:
  - `l515_pc_xyzs_l515` (262, 10000, 3) float32 and `l515_pc_rgbs` (262, 10000, 3) float32: coloured point clouds from an L515 depth camera. There are no RGB images.
  - `gripper_xyzs_l515` (262, 3), `gripper_quats_l515` (262, 4): a gripper pose in the camera frame.
  - `pyft_xyzs_l515` (262, 3), `pyft_quats_l515` (262, 4): pose of the force-sensing tool in the camera frame.
  - `pyft_fs` (262, 3) and `pyft_ts` (262, 3): force and torque. In this demo |F| percentiles 50 / 90 / 99 / max are 0.69 / 11.92 / 13.29 / 13.62 (assumed N).
  - `angler_widths` (262,): 0.008 to 0.048, likely a gripper width in m (not documented).
  - `a` (252, 20) int32 and `o` (252, 1) int32: these hold frame indices (first row of `a` is 1 to 20, `o` starts 0, 1, 2), not action values.
- The frame rate is not stored in the parts I read.
- What I saw (rendered point clouds, frames 0, 90, 180, 261): a yellow-green zucchini, a blue Y-shaped tool on its surface during the stroke, and a grey object at one end. In the last frame a pale peeled band runs along the vegetable. No human hand or robot arm is visible in the cropped cloud, so I could not tell who holds what. The vegetable identity rests mainly on the file name: cucumber and zucchini look alike at this resolution.
- Verdict: **peeling but not cucumber** (zucchini, per the file name and the look of the cloud).
- Contact sheet: `media/dataset_audit/forcemimic.jpg`.

## Which datasets are usable for a bimanual Franka cucumber peeling project, and how

Verified real cucumber peeling: RDP, ForceVLA, henryyiw (the 120-episode set, plus frames from the raw `_new` set), JensenYuan, VibeCuisine DAgger, tompollok, AgiBot task 550 (cucumber subset), Jefferzn, FoAR sample. Peeling but not cucumber: ForceMimic (zucchini). Could not verify: How to Peel (29 GB of pickled data, knife, no sample). Not usable as peeling data: `siyitian/peel_cucumber` (chip pick-and-place).

None of them is a bimanual Franka. They split into three hold strategies:

1. **Second arm holds the cucumber** (the layout we want): AgiBot task 550 (374 cucumber episodes, left arm pins the cucumber on the board, right arm peels) and Jefferzn (1 episode, in-air).
2. **A human holds the cucumber**: RDP.
3. **A fixture holds the cucumber**: ForceVLA (metal block), henryyiw and JensenYuan (motorised spindle), VibeCuisine and tompollok (blue clamp on a rail, blade mounted on the effector), FoAR (yellow blocks).

### Cross-embodiment pretraining

- **AgiBot task 550, cucumber subset**: the best structural match. Two 7-DoF arms, a holding arm and a peeling arm, end-effector poses for both arms, and sub-task segment frame ranges (grasp, pick peeler, peel, tidy, place, release). Use `episode_vegetables.csv` to keep the 374 cucumber episodes, or keep all 863 for a general peeling prior. The end-effector xyz + quaternion per arm can be retargeted to two Franka arms in task space. Check the card's research-only terms first.
- **ForceVLA** (43 episodes, single arm, grasped Y-peeler, 30 Hz) and **FoAR** (only 3 episodes in the public sample): useful for the peeling arm alone, with force.
- **henryyiw + JensenYuan** (120 + 156 + 61 episodes, if the `_new` set and Jensen do not overlap, which I did not check): many strokes on one rig, with GelSight tactile. They store joint state and a 6D TCP pose. Note that `actions` equals the current state and must be shifted. No licence is stated, so ask the authors before redistributing anything.
- **VibeCuisine / tompollok** (188 + 36 episodes): the blade is mounted on a low-cost servo arm, not grasped, so the tool interaction differs from a Franka holding a peeler. Joint-space only, no force. They may still help a vision encoder learn what "peeled" looks like.
- **RDP**: Flexiv arms, human-held cucumber. Good for the peeling arm's visuo-tactile policy, but the cucumber holder is a person.

### Force-profile reference

- **RDP**: 6-axis TCP wrench + gripper force + two tactile marker fields, at 12 fps in the downsampled zarr (24 fps raw). In the mini set the left |F| reaches about 12 N, with a median of about 3.3 N.
- **ForceVLA**: 6-axis external wrench in the base frame at 30 Hz. In ep19 |F| peaks near 10 N.
- **FoAR**: 6-axis F/T at 200 Hz (3 episodes). |F| median about 1.9 N, max about 25.8 N in scene_0002.
- **ForceMimic** (zucchini, 139 demos): tool-side force and torque with tool and gripper poses. In demo_03_shave_00 |F| reaches about 13.6 N. It is a different vegetable, and the 9 GB file would have to be read by range requests or downloaded in full.
- **AgiBot**: joint currents only, a weak proxy.
- henryyiw / JensenYuan: GelSight images only.

These give a realistic stroke force band for setting impedance and safety limits in the simulation. Treat the numbers as rough. Sensor frames and units are not all documented.

### Format reference

- **RDP zarr** (schema above) is the target layout for the main build. Keep `meta/episode_ends`, the 9-D TCP pose (xyz + 6D rotation), 6-D wrench, gripper width/force, 240x320 uint8 images and blosc compression. Fix two things the mini file shows: store `timestamp` as float64, and keep `target` row-aligned with the other arrays (or drop it). For a bimanual Franka, fill both `left_robot_*` and `right_robot_*` with real data (in RDP the right arm is static) and widen `action` beyond the 4-D single-arm xyz + width.
- **LeRobot v3.0** (AgiBot, henryyiw, VibeCuisine, tompollok, Jefferzn, siyitian) and **v2.1** (ForceVLA, JensenYuan) are the formats for loading the external data. The tools in `tools/audit/` read both without the lerobot package.
