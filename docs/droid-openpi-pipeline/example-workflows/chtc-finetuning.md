---
layout: default
title: CHTC Fine-tuning
parent: Example Workflows
grand_parent: DROID/OpenPI Lab Pipeline
nav_order: 3
permalink: /example-workflows/chtc-finetuning/
---

# CHTC Fine-tuning

Use this workflow to fine-tune Pi0.5-DROID on the converted lab dataset.

## Access Point

```bash
ssh <netid>@ap2002.chtc.wisc.edu
```

Clone or update OpenPI in your CHTC home directory:

```bash
cd ~
git clone https://github.com/Wisc-HCI/openpi.git
cd openpi
```

## Package OpenPI

`pack_openpi.sh` stages a `uv` binary and creates `openpi.tar.gz` next to the repo:

```bash
cd ~/openpi
./pack_openpi.sh
ls -lh ../openpi.tar.gz
```

## Stage Dataset

Large dataset archives should live in staging, not in the home directory:

```bash
scp realsense_droid_pick.tar.gz \
  <netid>@transfer.chtc.wisc.edu:/staging/y/yyi49/datasets/
```

The current submit file expects:

```text
osdf:///chtc/staging/y/yyi49/datasets/realsense_droid_pick.tar.gz
```

## Submit

From the CHTC home directory:

```bash
mkdir -p logs
condor_submit openpi/pi05_droid_finetune.sub
```

Current training entry point inside `run_pi05_droid_finetune.sh`:

```bash
uv run scripts/train.py pi05_droid_finetune \
  --exp-name=realsense_droid \
  --checkpoint-base-dir "$PWD/checkpoints" \
  --batch-size=2 \
  --num-train-steps=40 \
  --save-interval=5000 \
  --keep-period=None \
  --ema-decay=None \
  --no-wandb-enabled
```

`--num-train-steps=40` is a smoke test setting. For a formal run, increase it or remove the override so the config value is used.

## Monitor

```bash
condor_q
condor_tail <cluster.proc>
condor_tail -stderr <cluster.proc>
condor_q -hold
condor_q -better-analyze <cluster_id>
condor_history <cluster_id>
```

## Retrieve Checkpoints

The current job packages outputs as `checkpoints.tar.gz` and sends them to `/staging/y/yyi49/`.

```bash
rsync -avh --progress \
  <netid>@transfer.chtc.wisc.edu:/staging/y/yyi49/checkpoints.tar.gz \
  /home/kindred/Desktop/repo/openpi/
```

Then unpack on the 4090 workstation:

```bash
cd /home/kindred/Desktop/repo/openpi
tar -xzf checkpoints.tar.gz
find checkpoints -maxdepth 4 -type d | sort | tail
```
