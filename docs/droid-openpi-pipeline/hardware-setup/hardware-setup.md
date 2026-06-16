---
layout: default
title: Hardware Setup
parent: DROID/OpenPI Lab Pipeline
nav_order: 2
has_toc: false
permalink: /hardware-setup/
---

# Hardware Setup

This page is not a from-scratch shopping or assembly guide. It records the hardware assumptions for our existing lab pipeline.

## Required Hardware

| Hardware | Purpose | Current lab value |
| --- | --- | --- |
| Franka Panda | Executes arm actions from the policy. | Left Robot IP `192.168.4.3` |
| NUC | Runs DROID low-level robot server and Polymetis. | NUC IP `192.168.4.4` |
| Control laptop | Runs DROID GUI, cameras, Quest teleop, and rollout client. | Laptop IP `192.168.4.6` |
| External RealSense | Policy external camera input. | Serial `827312070419` |
| Wrist RealSense | Policy wrist camera input. | Serial `939622075130` |
| Quest controller | Teleoperation for demonstration collection. | Right controller by default |
| Tesollo gripper | Optional task gripper. | Control from laptop-side Open-Teach adapter |

## Safety Checks

Before launching robot control:

1. Clear the robot workspace.
2. Confirm the emergency stop is reachable.
3. In Franka Desk, unlock joints and activate FCI.
4. Confirm the NUC and laptop are on the robot network.
5. Confirm the camera mount is stable and the wrist cable is not in the motion path.

Running `check_nuc_control.py --launch` is not a passive connectivity check. It asks the NUC server to launch the controller and robot client, so brief robot motion can occur during controller handoff.

## Camera Checks

The DROID rollout client expects one external RealSense image and one wrist RealSense image. Current IDs are configured in the DROID checkout:

```bash
/path_to_your_droid_repo/droid/misc/parameters.py
```

The first rollout step writes `robot_camera_views.png`. Inspect it before trusting a policy run.
