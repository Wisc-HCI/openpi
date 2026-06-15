---
layout: default
title: DROID/OpenPI Lab Pipeline
nav_order: 1
has_toc: false
---

# DROID/OpenPI Lab Pipeline

This guide documents the lab workflow for using the current DROID hardware with OpenPI policies. It follows the same high-level structure as the official DROID docs, but it is specific to our setup:

1. Use DROID teleoperation to collect RealSense demonstrations.
2. Convert and publish the dataset for lab use.
3. Fine-tune `pi05_droid_finetune` from this OpenPI repo.
4. Start a policy server on the lab 4090 workstation.
5. Start the low-level DROID robot server on the NUC.
6. Run the VLA rollout client on the control laptop.

If you only want to run the existing policy and do not want to collect data or fine-tune, skip the data collection and CHTC sections. Start with the 4090 policy server, then the NUC server, then the control laptop rollout.

## System Roles

| Machine | Role | Current default |
| --- | --- | --- |
| 4090 workstation | Runs OpenPI policy inference server. | `/home/kindred/Desktop/repo/openpi`, port `8000` |
| NUC | Runs the DROID low-level Franka/Polymetis server. | `hcilab@192.168.4.4`, port `4242` |
| Control laptop | Runs DROID GUI, RealSense cameras, teleoperation, and rollout client. | `/home/kindred/Desktop/repo/droid` |
| CHTC Access Point | Submits fine-tuning jobs. | `ap2002.chtc.wisc.edu` |
| CHTC Transfer host | Moves large datasets/checkpoints through staging. | `transfer.chtc.wisc.edu`, `/staging/y/yyi49` |

## Guides

1. [Hardware Setup](./hardware-setup/hardware-setup.md)
2. [Software Setup](./software-setup/software-setup.md)
3. [Example Workflows](./example-workflows/example-workflows.md)

