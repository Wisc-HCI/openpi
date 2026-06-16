---
layout: default
title: CHTC Fine-tuning
parent: Example Workflows
grand_parent: DROID/OpenPI Lab Pipeline
nav_order: 3
permalink: /example-workflows/chtc-finetuning/
---

# CHTC Fine-tuning

Use this workflow to fine-tune Pi0.5-DROID on a lab dataset stored on Hugging Face and staged through CHTC `/staging`.

## Access Point

Log in to the CHTC access point. Use the access point for editing files, preparing submit scripts, and submitting jobs.

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
cd ~
bash openpi/pack_openpi.sh
```

## Download and Stage Dataset from Hugging Face

Large datasets should be downloaded and staged through CHTC's transfer host, not the access point.

Log in to the transfer host:

```bash
ssh <netid>@transfer.chtc.wisc.edu
```

Create a workspace and install the Hugging Face download package:

```bash
python -m pip install --user huggingface_hub
```

On your hugging face profile, navigate to settings -> Access Tokens, create a new write or read token, and use that token to login:

```bash
hf auth login
```

Paste your Hugging Face access token when prompted.

Download the dataset from Hugging Face:

```bash
cd /staging/<netID> #You may contact CHTC if you do not have quota. My directory is at /staging/y/yyi49, FYI
mkdir datasets
hf download peopleandrobots/realsense_pi05_pick --repo-type dataset --local-dir datasets/realsense_droid_pickup
```

Compress the downloaded dataset into a single archive:

```bash
cd datasets
tar -czf realsense_droid_pickup.tar.gz realsense_droid_pickup
```

The current submit file expects the dataset archive at:

```text
osdf:///chtc/staging/<netID>/datasets/realsense_droid_pickup.tar.gz
```

## Submit Fine-tuning Job

On the access point terminal(ap2002), submit from your CHTC home directory:

```bash
cd ~
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

`--num-train-steps=40` is a smoke test setting. For a formal run, increase it to 40-50k steps

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

From the 4090 workstation:

```bash
rsync -avh --progress \
  <netid>@transfer.chtc.wisc.edu:/staging/<netid>/checkpoints.tar.gz \
  /path_to_your_openpi_repo/
```

Then unpack on the 4090 workstation:

```bash
cd /path_to_your_openpi_repo/
tar -xzf checkpoints.tar.gz
find checkpoints -maxdepth 4 -type d | sort | tail
```