---
layout: default
title: Software Setup
parent: DROID/OpenPI Lab Pipeline
nav_order: 3
has_toc: false
permalink: /software-setup/
---

# Software Setup

This setup is split by machine. Currently we are using 3 computers. Keep OpenPI policy dependencies separate from DROID robot-control dependencies.

## 4090 Workstation: OpenPI

```bash
cd /home/kindred/Desktop/repo/openpi
uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The policy server is started from this repo:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

For a fine-tuned checkpoint, use the fine-tuning config and local checkpoint directory:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid_finetune \
  --policy.dir=/path/to/checkpoint_step
```

## Control Laptop: DROID + OpenPI Client

Install the lightweight OpenPI client into the DROID environment:

```bash
conda activate droid
cd /home/kindred/Desktop/repo/openpi/packages/openpi-client
pip install -e .
pip install tyro
```

The control laptop uses the DROID checkout:

```bash
cd /home/kindred/Desktop/repo/droid
```

Key scripts:

| Script | Purpose |
| --- | --- |
| `scripts/collect_openpi_realsense.py` | Collect OpenPI-compatible RealSense DROID demonstrations. |
| `scripts/main.py` | Run VLA policy rollouts against the 4090 policy server. |
| `scripts/tests/check_nuc_control.py` | Check laptop to NUC to Franka communication. |

## NUC: DROID Low-level Server

On the NUC:

```bash
ssh hcilab@192.168.4.4
cd /home/hcilab/Desktop/github/droid
bash scripts/server/launch_server.sh
```

The server binds `tcp://0.0.0.0:4242`. The control laptop connects through DROID `ServerInterface`.

## Hugging Face CLI

Use Hugging Face as the lab dataset registry:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf auth login
hf auth whoami
```

For this lab workflow, do not put credentials in submit files or committed scripts.
