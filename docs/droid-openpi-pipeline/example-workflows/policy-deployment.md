---
layout: default
title: Policy Deployment
parent: Example Workflows
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

## Rollout-owned steering

With the JAX Pi0/Pi0.5 server, set the competing instruction on the rollout command:

```bash
cd /home/hci-lab/repos/droid
python scripts/main.py \
  --remote_host=192.168.4.9 \
  --remote_port=8000 \
  --external_camera=left \
  --max_timesteps=500 \
  --competing_command="pick up the green block" \
  --steering_mode=fixed \
  --steering_scale=1.0
```

Then enter the positive instruction at `Enter instruction:`, for example `pick up the blue block`.
The client sends both instructions and all steering settings with every query.
Update both repositories and restart the server once. After that, keep the server
running and change settings only in `main.py`:

| Mode | Client flags | Weight |
| --- | --- | --- |
| Fixed | `--steering_mode=fixed --steering_scale=1.0` | Always 1 |
| Decay | `--steering_mode=time_decay --steering_scale=1.0 --steering_decay=0.9` | 1, 0.9, 0.81, ... per query |
| Belief | `--steering_mode=belief --steering_scale=0.0 --belief_lambda=2.0 --belief_temperature=0.1` | First chunk 0; subsequently 2 × b_negative |
| Off | `--steering_mode=off` | 0, no observer |

Values are illustrative, not tuned recommendations. `steering_scale` independently
sets the FIRST chunk weight in belief mode (default 1.0). Temperature must be
explicitly positive in belief mode. Other belief defaults: 8 residual samples, tau
range [0.3, 0.7], 8 physical action dimensions, lambda 1.0. Configure them with
`--belief_samples`, `--belief_tau_min`, `--belief_tau_max`, `--belief_action_dims`,
and `--belief_lambda`. Hyphenated option names are also accepted.

Old **server CLI** `--negative-prompt[-file]`, `--guidance-*`, and `--belief-*`
options have been removed. Historical LIBERO shell workflows using those options
must be migrated separately; they are not silently mapped to new settings.
Server startup now specifies checkpoint/environment, port, default task prompt,
and optional recording. Library-level APIs remain available for offline callers;
explicit request settings never inherit those defaults.

Steering requires JAX Pi0/Pi0.5. The client verifies protocol support before robot
initialization and checks the server's configuration acknowledgement before
executing returned actions. Empty or identical positive/competing instructions
are rejected when steering is enabled.

New rollout IDs, configuration changes, instruction changes, and resets clear
belief and decay state; old commands are discarded. The observer still uses issued
commands, not measured trajectories. Weights remain resident; first-use JAX
compilation can still add latency. Use one active rollout per server at a time.

The rollout CSV records mode and parameters even without `--record_tracking`.
Optional tracking logs record acknowledged settings in each query response, not
in server startup metadata.
