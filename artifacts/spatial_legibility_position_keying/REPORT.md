# Absolute-position keying audit

This audit reads the existing checkpoint-39999 manifest and results without modifying them. Distances compare
each held-out grasp-onset end-effector position with all fine-tuning target positions.

| Condition | True | VLA prediction | Same XYZ | Opposite XYZ | XYZ 1-NN | Same x | Opposite x | x 1-NN | Same y | Opposite y | y 1-NN |
|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---|
| C0_natural_direct | left | right | 0.0216 | 0.0530 | left | 0.0096 | 0.0090 | right | 0.0021 | 0.0421 | left |
| C0_natural_direct | right | left | 0.0113 | 0.0517 | right | 0.0025 | 0.0081 | right | 0.0013 | 0.0135 | right |
| C1_mild_exaggeration | left | left | 0.0306 | 0.0756 | left | 0.0039 | 0.0146 | left | 0.0087 | 0.0657 | left |
| C1_mild_exaggeration | right | left | 0.0125 | 0.0729 | right | 0.0030 | 0.0156 | right | 0.0018 | 0.0181 | right |
| C2_strong_exaggeration | left | left | 0.0193 | 0.0581 | left | 0.0065 | 0.0121 | left | 0.0012 | 0.0473 | left |
| C2_strong_exaggeration | right | left | 0.0089 | 0.0781 | right | 0.0019 | 0.0125 | right | 0.0004 | 0.0127 | right |
| C3_extreme_exaggeration | right | right | 0.0308 | 0.0274 | left | 0.0077 | 0.0064 | left | 0.0015 | 0.0006 | left |
| C3_extreme_exaggeration | left | right | 0.0046 | 0.0524 | left | 0.0012 | 0.0010 | right | 0.0016 | 0.0371 | left |

## Summary

- A same-side training target is closer in 3D for **7/8** held-out targets; the exception is
  **C3_extreme_exaggeration right**. A 3D 1-nearest-target side classifier therefore has accuracy **0.875**.
- Using coordinate `grasp_x` alone, an opposite-side target is closer for **3/8** cases:
  **C0_natural_direct left, C3_extreme_exaggeration right, C3_extreme_exaggeration left**. The x-only 1-NN accuracy is **0.625**.
- Using coordinate `grasp_y` alone, an opposite-side target is closer for **1/8** cases:
  **C3_extreme_exaggeration right**. The y-only 1-NN accuracy is **0.875**.
- The VLA prediction agrees with the 3D nearest-target label in only **0.250** of cases.

In this robot coordinate frame, `grasp_y` separates the training left/right distributions much more strongly
than `grasp_x`; screen-horizontal direction should not be inferred from the field name alone. Crucially, both
C0 targets are closer to same-side training targets in 3D and in `grasp_y`. Therefore target-coordinate
nearest-neighbor keying does **not** explain the C0 Early-AUC of 0.075. The sole 3D opposite-nearest case is
C3-right, which the VLA actually classifies correctly, also opposing the proposed mechanism.

The x-only observation is weaker: C0-left is 0.6 mm closer in x to an opposite-side training target, and the
VLA gets it wrong. But x alone is not the physical left/right axis here, the difference is tiny, and the same
rule also predicts C3-right incorrectly while the VLA gets it right. This endpoint audit does not rule out
**visual pixel-position keying**, which would require first-frame block annotations or detections rather than
robot Cartesian grasp coordinates.
