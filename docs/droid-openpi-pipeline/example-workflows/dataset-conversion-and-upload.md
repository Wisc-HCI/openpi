---
layout: default
title: Dataset Conversion and Upload
parent: Example Workflows
grand_parent: DROID/OpenPI Lab Pipeline
nav_order: 2
permalink: /example-workflows/dataset-conversion-and-upload/
---

# Dataset Conversion and Upload

Convert the collected DROID/RealSense data into the LeRobot layout expected by `pi05_droid_finetune`.

## Dry Run

```bash
cd /home/kindred/Desktop/repo/openpi
uv run examples/droid/convert_realsense_droid_data_to_lerobot.py \
  --data-dir /home/kindred/Desktop/repo/droid/data \
  --repo-id Wisc-HCI/realsense_droid_pick \
  --skip-invalid \
  --dry-run
```

## Convert

```bash
uv run examples/droid/convert_realsense_droid_data_to_lerobot.py \
  --data-dir /home/kindred/Desktop/repo/droid/data \
  --repo-id Wisc-HCI/realsense_droid_pick \
  --skip-invalid \
  --overwrite
```

## Upload

```bash
uv run examples/droid/convert_realsense_droid_data_to_lerobot.py \
  --data-dir /home/kindred/Desktop/repo/droid/data \
  --repo-id Wisc-HCI/realsense_droid_pick \
  --skip-invalid \
  --overwrite \
  --push-to-hub \
  --private
```

You can also upload an already converted folder:

```bash
hf upload Wisc-HCI/realsense_droid_pick \
  ~/.cache/huggingface/lerobot/Wisc-HCI/realsense_droid_pick \
  . \
  --repo-type dataset
```

If you change the dataset repo id, update `repo_id` in `src/openpi/training/config.py` under `pi05_droid_finetune`.
