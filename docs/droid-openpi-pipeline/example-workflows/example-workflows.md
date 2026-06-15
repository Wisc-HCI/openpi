---
layout: default
title: Example Workflows
parent: DROID/OpenPI Lab Pipeline
nav_order: 4
has_children: true
has_toc: false
---

# Example Workflows

Use these workflows after hardware and software setup are complete.

1. [Teleoperation and Data Collection](./teleoperation-and-data-collection.md)
2. [Dataset Conversion and Upload](./dataset-conversion-and-upload.md)
3. [CHTC Fine-tuning](./chtc-finetuning.md)
4. [Policy Deployment](./policy-deployment.md)
5. [Troubleshooting](./troubleshooting.md)

## Fast Path: Run Existing Policy Only

If you do not need to collect data or fine-tune:

1. Start the OpenPI policy server on the 4090.
2. Start the DROID low-level server on the NUC.
3. Run `scripts/main.py` on the control laptop.
4. Enter the natural-language instruction when prompted.

