---
layout: default
title: Example Workflows
parent: DROID/OpenPI Lab Pipeline
nav_order: 4
has_children: true
has_toc: false
permalink: /example-workflows/
---

# Example Workflows

Use these workflows after hardware and software setup are complete.

1. [Teleoperation and Data Collection]({{ site.baseurl }}/example-workflows/teleoperation-and-data-collection/)
2. [Dataset Conversion and Upload]({{ site.baseurl }}/example-workflows/dataset-conversion-and-upload/)
3. [CHTC Fine-tuning]({{ site.baseurl }}/example-workflows/chtc-finetuning/)
4. [Policy Deployment]({{ site.baseurl }}/example-workflows/policy-deployment/)
5. [Troubleshooting]({{ site.baseurl }}/example-workflows/troubleshooting/)

## Fast Path: Run Existing Policy Only

If you do not need to collect data or fine-tune:

1. Start the OpenPI policy server on the 4090.
2. Start the DROID low-level server on the NUC.
3. Run `scripts/main.py` on the control laptop.
4. Enter the natural-language instruction when prompted.
