---
layout: default
title: Teleoperation and Data Collection
parent: Example Workflows
grand_parent: DROID/OpenPI Lab Pipeline
nav_order: 1
permalink: /example-workflows/teleoperation-and-data-collection/
---

# Teleoperation and Data Collection

Use this workflow only when collecting new demonstrations.

## Start the NUC Server

```bash
ssh hcilab@192.168.4.4
cd /home/hcilab/Desktop/github/droid
bash scripts/server/launch_server.sh
```

## Start the DROID Collection GUI

On the control laptop:

```bash
cd /home/kindred/Desktop/repo/droid
conda activate droid
python scripts/collect_openpi_realsense.py
```

This script creates `RobotEnv()`, uses Quest `VRPolicy`, and saves data with `save_data="openpi"`.

## Teleoperation Notes

1. Plug the Quest controller into the control laptop.
2. Accept USB debugging inside the headset.
3. Keep the controller visible to the headset cameras.
4. Use the right controller unless the DROID controller code is changed.
5. Hold the side grip button to apply motion.
6. Use the front trigger for gripper control.
7. Press the joystick to redefine the controller forward direction.

Start with `Practice` in the GUI before saving real trajectories.

## Saved Data Layout

```text
/home/kindred/Desktop/repo/droid/data/
  success/
    YYYY-MM-DD/
      episode_name/
        trajectory.h5
        metadata_openpi.json
        recordings/
          MP4/
            camera_id.mp4
  failure/
    YYYY-MM-DD/
      ...
```

The task text entered in the GUI is written into `metadata_openpi.json` and becomes the policy prompt after conversion.
