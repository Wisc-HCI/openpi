---
layout: default
title: Policy Deployment
parent: Example Workflows
grand_parent: DROID/OpenPI Lab Pipeline
nav_order: 4
permalink: /example-workflows/policy-deployment/
---

# Policy Deployment

Use this workflow to run an OpenPI policy on the real DROID hardware.

## 1. Start Policy Server on 4090

Public Pi0.5-DROID checkpoint:

```bash
cd /home/kindred/Desktop/repo/openpi
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

Fine-tuned checkpoint:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid_finetune \
  --policy.dir=/home/kindred/Desktop/repo/openpi/checkpoints/pi05_droid_finetune/realsense_droid/40
```

## 2. Start Low-level Server on NUC

```bash
ssh hcilab@192.168.4.4
cd /home/hcilab/Desktop/github/droid
bash scripts/server/launch_server.sh
```

## 3. Check NUC From Control Laptop

```bash
cd /home/kindred/Desktop/repo/droid
conda activate droid
python scripts/tests/check_nuc_control.py --host 192.168.4.4
```

If the ZeroRPC server is reachable but the robot client is not launched:

```bash
python scripts/tests/check_nuc_control.py --host 192.168.4.4 --launch
```

## 4. Run VLA Rollout

Franka-only:

```bash
python scripts/main.py \
  --remote_host=<4090_ip> \
  --remote_port=8000 \
  --external_camera=left \
  --max_timesteps=600 \
  --open_loop_horizon=8
```

With Tesollo gripper:

```bash
python scripts/main.py \
  --remote_host=<4090_ip> \
  --remote_port=8000 \
  --external_camera=left \
  --enable_tesollo_gripper \
  --openteach_root=/home/kindred/Desktop/repo/Open-Teach
```

The script prompts for `Enter instruction:` and saves rollout results under `results/`.
