# 实验结果对比

## 模型对比表格 (最高ACC)

| 方法 | cv F1 | cv ACC | test1 F1 | test1 ACC | test2 F1 | test2 ACC | test3 F1 | test3 ACC |
|------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| Baseline (v0) | 0.7366 | 0.7376 | 0.7956 | 0.7956 | 0.7450 | 0.7476 | 0.8638 | 0.8645 |
| Robust v1 (Dropout) | 0.7491 | 0.7516 | 0.8239 | 0.8248 | 0.7609 | 0.7621 | 0.8910 | 0.8945 |
| Robust v4 | 0.7722 | 0.7732 | 0.8302 | 0.8297 | 0.7832 | 0.7840 | 0.8907 | 0.8921 |
| Robust v5 | 0.7427 | 0.7447 | 0.8113 | 0.8127 | 0.7723 | 0.7767 | 0.8939 | 0.8993 |
| **P-RMF V2 (VAE)** | **0.7586** | **0.7593** | **0.8348** | **0.8345** | **0.7693** | **0.7718** | **0.8995** | **0.9029** |



### P-RMF V2 最佳结果详情

| Split | F1 | ACC | Val | 文件 |
|-------|---:|----:|----:|------|
| cv | 0.7586 | 0.7593 | 0.6317 | `cv_features:Baichuan-13B-Base-UTT+chinese-hubert-large-UTT+c...` |
| test1 | 0.8348 | 0.8345 | 0.6431 | `test1_features:Baichuan-13B-Base-UTT+chinese-hubert-large-UT...` |
| test2 | 0.7693 | 0.7718 | 0.6170 | `test2_features:Baichuan-13B-Base-UTT+chinese-hubert-large-UT...` |
| test3 | 0.8995 | 0.9029 | 79.9783 | `test3_features:Baichuan-13B-Base-UTT+chinese-hubert-large-UT...` |