---
layout: default
title: Dataset Conversion and Upload
parent: Example Workflows
grand_parent: DROID/OpenPI Lab Pipeline
nav_order: 2
permalink: /example-workflows/dataset-conversion-and-upload/
---

# Dataset Conversion and Upload

On your control laptop, convert the collected DROID/RealSense data into the LeRobot layout expected by `pi05_droid_finetune`. In this section assume we collect trajectories for picking up task.

## Dry Run

```bash
cd /path_to_your_openpi_repo
uv run examples/droid/convert_realsense_droid_data_to_lerobot.py \
  --data-dir /path_to_your_droid_repo/data \
  --repo-id peopleandrobots/realsense_droid_pickup \
  --skip-invalid \
  --dry-run
```

## Convert

```bash
uv run examples/droid/convert_realsense_droid_data_to_lerobot.py \
  --data-dir /path_to_your_droid_repo/data \
  --repo-id peopleandrobots/realsense_droid_pickup \
  --skip-invalid \
  --overwrite
```

## Upload

```bash
uv run examples/droid/convert_realsense_droid_data_to_lerobot.py \
  --data-dir /path_to_your_droid_repo/data \
  --repo-id peopleandrobots/realsense_droid_pickup \
  --skip-invalid \
  --overwrite \
  --push-to-hub \
  --private
```

If you change the dataset repo id, update `repo_id` in `src/openpi/training/config.py` under `pi05_droid_finetune`.
