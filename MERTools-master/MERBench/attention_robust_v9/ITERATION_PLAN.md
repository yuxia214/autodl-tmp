# AttentionRobust V9 Iteration Plan

## Objective
Improve MER2023 challenge-aligned metrics:
- test1/test2 Combined = F1 - 0.25 * MSE
- test3 Weighted-F1

## Key Architecture Updates
1. Quality-aware fusion
- Fuse uncertainty weights with a learned quality score per modality.
- Quality score conditions on latent mean/std, reconstruction error, and observed flag.

2. Cross-modal imputation
- Randomly mask one/two modalities during training.
- Predict missing modality latent representation from remaining modalities.

3. Teacher-student consistency
- Teacher: clean path.
- Student: noisy + masked + imputed path.
- Constrain classification distribution and valence regression consistency.

## Directional Experiment Set
Use `run_v9_directional.sh` with 4 configs:
- base
- imp_strong
- cons_strong
- high_corrupt

## Selection Rule
Prioritize:
1. test2 Combined
2. test1 Combined
3. test3 F1
