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
cd /path_to_your_openpi_repo/
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
  --policy.dir=/path_to_your_openpi_repo/checkpoints/pi05_droid_finetune/realsense_droid/<steps>
```

## 2. Start Low-level Server on NUC

```bash
ssh hcilab@192.168.4.4
cd /home/hcilab/Desktop/github/droid
bash scripts/server/launch_server.sh
```

## 3. Run VLA Rollout

Franka-only:

```bash
python scripts/main.py   --remote_host=192.168.4.5   --remote_port=8000   --external_camera=left   --max_timesteps=500
```


The script prompts for `Enter instruction:` and saves rollout results under `results/`.
