---
layout: default
title: Troubleshooting
parent: Example Workflows
nav_order: 5
---

# Troubleshooting

## Control Laptop Cannot Reach Policy Server

Check that the server is still running on the 4090 and that the control laptop can reach port `8000`:

```bash
python -c "import socket; s=socket.create_connection(('<4090_ip>', 8000), timeout=3); s.close(); print('reachable')"
```

## NUC TCP Works but Robot State Fails

This usually means the ZeroRPC server is alive but the NUC-local Franka/Polymetis client is not running or crashed. Check the NUC terminal logs and then launch through:

```bash
python scripts/tests/check_nuc_control.py --host 192.168.4.4 --launch
```

## Cameras Missing or Swapped

Check the camera serials in:

```bash
/home/kindred/Desktop/repo/droid/droid/misc/parameters.py
```

Then inspect `robot_camera_views.png` after the first rollout step.

## CHTC Job Cannot Find `uv`

Run `./pack_openpi.sh` before submitting and confirm `openpi/bin/uv` is inside the tarball.

## CHTC `ImportError: libGL.so.1`

Use headless OpenCV dependencies in the job environment. GUI OpenCV wheels are a common cause on headless GPU nodes.

## Tyro Boolean Flag Error

Use:

```bash
--no-wandb-enabled
```

Do not use:

```bash
--wandb-enabled=false
```

## Robot Moves in the Wrong Direction

Check action semantics before changing networking or camera code. The DROID runtime names the first seven action dimensions `joint_velocity`, but the deployment path treats them as normalized joint deltas. Training labels must match that execution meaning.

