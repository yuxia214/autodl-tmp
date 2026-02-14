# AttentionRobust V8 Iteration Plan

## Goal
Improve challenge metrics on MER2023, especially Combined on test1/test2 while preserving test3 F1.

## Architecture Changes
- Dual-path fusion:
  - Main path: uncertainty fusion + proxy cross-modal attention.
  - Residual path: modality experts weighted by learned reliability.
- Regularization:
  - Modality agreement regularization on latent means.
  - Weight consistency regularization between reliability weights and fusion weights.
- Keep V7 improvements:
  - emotion-guided valence prior
  - valence consistency regularization
  - feature noise augmentation

## Initial Training Setup
Use `train_v8.sh` defaults, then compare to best V7:
- test1/test2: Combined = F1 - 0.25 * MSE
- test3: Weighted F1
