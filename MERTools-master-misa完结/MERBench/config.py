# *_*coding:utf-8 *_*
import os
import sys

# === 1. 标签文件所在目录 (保持原状) ===
# 存放 label-6way.npz 的地方
LABEL_ROOT = '/root/autodl-tmp/MERTools-master/MERBench/dataset/mer2023-dataset-process'

# === 2. 特征文件所在目录 (修改这里) ===
# 存放 chinese-hubert-large-UTT 等文件夹的地方
FEATURE_ROOT = '/root/autodl-tmp/features'

DATA_DIR = {
    'MER2023': LABEL_ROOT,
}

PATH_TO_RAW_AUDIO = {
    'MER2023': os.path.join(FEATURE_ROOT, 'audio'), # 假设原始音频也在 features 下，如果不是可不改
}

PATH_TO_RAW_FACE = {
    'MER2023': os.path.join(FEATURE_ROOT, 'openface_face'),
}

PATH_TO_TRANSCRIPTIONS = {
    'MER2023': os.path.join(LABEL_ROOT, 'transcription.csv'),
}

# [关键修改] 让代码去 features 目录找特征
PATH_TO_FEATURES = {
    'MER2023': FEATURE_ROOT,
}

# [保持不变] 让代码去原目录找标签
PATH_TO_LABEL = {
    'MER2023': os.path.join(LABEL_ROOT, 'label-6way.npz'),
}

# === 工具路径 (Tools) ===
PATH_TO_PRETRAINED_MODELS = '/root/autodl-tmp/MERTools-master/MERBench/tools'
PATH_TO_OPENSMILE = os.path.join(PATH_TO_PRETRAINED_MODELS, 'opensmile-2.3.0')
PATH_TO_FFMPEG = '/usr/bin/ffmpeg'
PATH_TO_NOISE = os.path.join(PATH_TO_PRETRAINED_MODELS, 'musan/audio-select')

# === 结果保存路径 ===
# === 结果保存路径 ===
# [修改建议] 获取 config.py 所在的绝对目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 将保存路径强制指向当前目录下的 saved-trimodal，确保它是绝对路径
SAVED_ROOT = os.path.join(CURRENT_DIR, 'saved-trimodal')

# 下面的保持不变，它们会自动继承上面的绝对路径
DATA_DIR_SAVE = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')