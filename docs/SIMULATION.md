# How the simulation works

This page explains exactly what is simulated, what is a model assumption, and what is not simulated at all.

## The cell

- Two Franka Emika Panda arms (7 joints each, plus the Franka Hand), taken unchanged from MuJoCo Menagerie
  (`franka_emika_panda`, pinned commit `4d038b3`, Apache-2.0). Bases side by side, 0.5 m apart, both facing +x.
- **Left arm**: holds the cucumber by one end. The cucumber lies across the front of both robots, tip toward the
  right arm. Rolling the cucumber about its own axis uses joint 7 only, which gives about 320 degrees of roll.
- **Right arm**: a Y-style peeler is mounted rigidly in the hand (like a printed adapter on the flange). The tool
  centre point (TCP) is the blade edge. A 6-axis force/torque site sits at the flange.
- Cameras: an external camera in front of the cell and a wrist camera on a side bracket of the right hand, both
  96 x 128 pixels for the policy. Videos use a 640 x 480 render of the detailed meshes.

The layout was not guessed. Three layouts were tried and checked by a search
(`scripts/dev/search_layout.py`, `scripts/dev/search_cross_layout.py`): side by side with the cucumber pointing forward
(the forearms collide near the gripper end), facing arms (the left hand cannot point forward at that height:
joint 6 would sit on its limit), and the final cross layout, which is collision free along every stroke with at
least 0.73 rad (left) and 0.79 rad (right) of joint margin.

## Control (same structure as the real robot)

```
policy or GELLO leader (24 Hz)  ->  TCP targets  ->  Cartesian impedance (480 Hz)  ->  joint torques
```

- Impedance law per arm: `tau = J^T (K e - D xdot) + gravity and Coriolis + null-space posture`, with the pose
  error clipped to 5 cm and 0.4 rad (as serl_franka_controllers and CRISP do on real Frankas).
- Gains: left (holder) 2200 N/m and 150 N m/rad, right (peeler) 1100/1100/900 N/m and 60 N m/rad. Damping is set
  for an effective arm mass of about 2.5 kg; unit-mass damping made the blade chatter in contact.
- GELLO path: a GELLO leader sends joint angles. The simulation turns them into a TCP target with forward
  kinematics and also uses them as the null-space posture. So GELLO, the scripted expert and the learned policies
  all go through the same controller.
- Force/torque reading: the external wrench on the tool, in the base frame, with the tool's own weight removed
  (like Franka's `O_F_ext_hat_K` or a tared wrist sensor), plus 0.05 to 0.25 N of sensor noise.

## The cucumber

Real cucumbers differ in length, thickness, bend and end shape, so every episode draws a cucumber from a class.
The **v2 classes follow real cucumber types** (`peel/shape.py`, `SHAPE_CLASSES_V2`):

| Class | Length | Diameter | Bend (inner arc per 10 cm) | Other |
|---|---|---|---|---|
| pickling (Kirby type) | 9 to 14 cm | 30 to 40 mm | 0 to 10 mm | short and stout |
| persian (mini) | 11 to 16 cm | 24 to 32 mm | 0 to 8 mm | slim |
| slicing | 18 to 25 cm | 42 to 56 mm | 0 to 10 mm | thick |
| english (hothouse) | 28 to 36 cm | 38 to 48 mm | 0 to 6 mm | long, often a narrow stem end (neck 10 to 30 %) |
| crooked (Class II) | 18 to 30 cm | 38 to 52 mm | 10 to 20 mm | gradual bend along the whole length |
| necked | 18 to 30 cm | 38 to 52 mm | 0 to 10 mm | stem end 25 to 40 % narrower ("bottle neck") |
| realistic | mix of the four market types | | 75 % Class I, 25 % Class II | half of them necked |
| very crooked (stress test only) | 18 to 28 cm | 38 to 50 mm | 20 to 28 mm | out of grade |

Where the numbers come from:

- **Bend** uses the official grading unit of the UNECE cucumber standard FFV-15
  (https://unece.org/fileadmin/DAM/15Cucumbers_2010.pdf): Class I must be "practically straight (maximum height of
  the inner arc: 10 mm per 10 cm of length)", Class II may be "slightly crooked" up to 20 mm per 10 cm, and more
  crooked ones must be packed separately. The bend is a gradual, constant-curvature arc along the whole cucumber.
  Held at one end, the tip of such a cucumber sits about 4 times the arc height away from the gripper axis, which is
  why a "practically straight" 30 cm cucumber can still be 10 cm off axis at the tip.
- **Lengths**: Persian 4 to 6 in, Kirby 3 to 6 in, English about 14 in
  (https://www.webstaurantstore.com/article/1063/types-of-cucumbers.html). Gherkins (1 to 3 in) are too short to
  hold and peel and are left out.
- **Diameters** are typical market values, not from a standard.
- **Mass** follows from the volume and a density of 0.96 g/cm3 (+-10 %), giving about 70 to 110 g for pickling and
  Persian, 300 to 450 g for slicing and 450 to 600 g for English cucumbers.

The **v1 classes** (mini, standard, long, thick, tapered, curved, mixed: straighter, thinner shapes) were the first
experiments. They are kept so the first results stay reproducible.

The body is a chain of 16 overlapping capsules (12 in v1) along the curved, tapered, necked centre line (MuJoCo
capsules cannot taper or bend). Per episode the simulation also randomises the grasp (3 mm offset, 2.5 degree
tilt), friction (0.15 to 0.35), firmness (contact time constant 8 to 14 ms) and skin toughness (below). Bumpy or
ridged skin (Kirby, Japanese types) and oval cross-sections are not modelled.

## The skin model (the most important assumption)

No mainstream simulator can cut a thin skin off a vegetable with a blade, and the research found no published
sim-to-real peeling work. So the skin is **not** simulated as a material. It is a grid of cells (5 mm along the
cucumber, 10 degrees around it) on the body surface. At 240 Hz, for each blade contact, a cell is removed only if
all of these hold at the contact:

1. normal force above the cut threshold (2.5 to 4.5 N per episode); below it the blade skids over the skin,
2. the blade edge dips into the surface by at least 2 degrees and at most 35 degrees (rake angle),
3. the blade moves along the cucumber with its edge leading, faster than 4 mm/s.

The strip width grows with force: 8 mm at the cut threshold, +1.2 mm per extra newton, capped at 20 mm. While
cutting, a resisting force of 160 to 280 N per metre of strip width acts against the stroke (on the blade and,
equal and opposite, on the cucumber), so the force sensor feels cutting.

If the force averaged over about 30 ms exceeds the gouge threshold (20 to 28 N per episode), or the edge is
steeper than 50 degrees, the cells are marked **gouged**: the skin is off but flesh was wasted.

Metrics: **coverage** = share of peelable cells removed (the gripped 2.5 cm and the rounded ends are excluded),
**gouge** = share of cells gouged. An episode counts as a success when coverage is at least 90 % and gouge is at
most 3 % and the peeler handle never hit the cucumber.

All thresholds are model assumptions chosen to match the force levels reported in the papers (ForceMimic's demos
around 10 N, damage above about 20 N). They are not measured cucumber mechanics.

## What is not simulated

- No fracture, no peel strip coming off, no juice, no slipping of the cucumber in the fingers (the grasp is rigid).
- Images are rendered on the CPU (this PC's GPU path is not available under WSL2) with low-poly collision meshes,
  so they look simpler than real camera images.
- Demos come from a scripted expert, not from people (see DATA.md).
