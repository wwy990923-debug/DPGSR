# VG-DP-GSR: Verification-Guided Dual-Path Group Sparse Representation

MATLAB implementation of VG-DP-GSR and VG-DP-GSR-ST for small-sample image classification.

## Requirements

- MATLAB R2018b or later
- No additional toolboxes required

## Quick Start

```matlab
run('demo_vgdpgsr.m')
```

Expected output on Yale B Subset 1 (7 train / 57 test per class):

```
Forward: ~70%   Reverse: ~77%   VG-DP-GSR: ~77%
```

## Files

| File | Description |
|------|-------------|
| `Forward.m` | Forward group-sparse representation solver (ADMM) |
| `Reverse.m` | Reverse group-sparse representation solver (ADMM) |
| `DPGSR_Fusion.m` | Dual-path fusion with hard-switch rule and verification scoring |
| `demo_vgdpgsr.m` | Demo script on Extended Yale B (cross-illumination) |
| `pcayaleb50.mat` | Extended Yale B dataset, PCA-50 features (38 classes, 64 images/class) |

## Method Overview

**VG-DP-GSR** consists of three components:

1. **Forward Path** — Group-sparse coding of test samples over the training dictionary
2. **Reverse Path** — Training samples reconstructed over the test batch, used as a verifier (not a classifier)
3. **Hard-Switch Fusion** — Verification score $v_j$ gates whether to trust forward or switch to reverse

**VG-DP-GSR-ST** extends this with verification-guided self-training: high-verification samples are added to the training dictionary with pseudo-labels.

## Parameters

| Parameter | VG-DP-GSR | VG-DP-GSR-ST | Description |
|-----------|:---------:|:------------:|-------------|
| $\lambda_1$ | 1.0 | 1.0 | Forward group-sparsity |
| $\lambda_2$ | 0.01 | 0.1 | Reverse group-sparsity |
| $\theta$ | 0.70 | 0.70 | Verification threshold |
| $\delta$ | 0.05 | 0.05 | Reverse margin threshold |
| $R_{\max}$ | -- | 2 | Max self-training rounds |

Parameters selected on Extended Yale B with PCA-50 and transferred directly to other datasets.

## Usage

```matlab
% Load your data: S (train, d×N), Sl (train labels, 1×N)
%                  T (test, d×M),  Tl (test labels, 1×M)
% nt = number of training samples per class

[forward_pred, acc, output] = DPGSR_Fusion(S, T, Sl, Tl, nt, ...
    lambda1, lambda2, 'adaptive', [], [], theta, delta);

% output.PF  — forward posterior probabilities
% output.PR  — reverse posterior probabilities
% output.D   — reverse coefficient matrix (for verification scores)
% acc        — hard-switch fusion accuracy
```

## Citation

If you use this code, please cite:

```
Wang, W.-Y., Chen, C.-K., & Qiao, X.-G.
Verification-Guided Dual-Path Group Sparse Representation with Self-Training.
```
