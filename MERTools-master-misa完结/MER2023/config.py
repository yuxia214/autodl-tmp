# *_*coding:utf-8 *_*
import os
import sys
import socket

############ For LINUX ##############

# [修正1] 使用你 find 命令找到的真实绝对路径
# 原始位置: /root/autodl-tmp/MERTools-master/MERBench/dataset/mer2023-dataset-process
REAL_DATA_ROOT = '/root/autodl-tmp/MERTools-master/MERBench/dataset/mer2023-dataset-process'

# 缺失模态增强参数
parser.add_argument('--modal_dropout_prob', type=float, default=0.2, help='训练时随机丢弃模态的概率')
parser.add_argument('--cross_recon_weight', type=float, default=0.5, help='跨模态重建损失权重')
DATA_DIR = {
    'MER2023': REAL_DATA_ROOT,
}

PATH_TO_RAW_AUDIO = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
}

PATH_TO_RAW_FACE = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
}

PATH_TO_TRANSCRIPTIONS = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription.csv'),
}

PATH_TO_FEATURES = {
    'MER2023': REAL_DATA_ROOT,  # 特征文件通常就在这个目录下
}

# [修正2] 确保指向真实存在的 label 文件
PATH_TO_LABEL = {
    'MER2023': os.path.join(REAL_DATA_ROOT, 'label-6way.npz'),
}

# 工具路径 (指向当前项目下的 tools)
PATH_TO_PRETRAINED_MODELS = '/root/autodl-tmp/MERTools-master/MERBench/tools'
PATH_TO_OPENSMILE = os.path.join(PATH_TO_PRETRAINED_MODELS, 'opensmile-2.3.0')
PATH_TO_FFMPEG = '/usr/bin/ffmpeg' # AutoDL 环境通常自带 ffmpeg，或指向项目内 tools
PATH_TO_NOISE = os.path.join(PATH_TO_PRETRAINED_MODELS, 'musan/audio-select')

# 保存路径
SAVED_ROOT = '/root/autodl-tmp/MERTools-master/MERBench/saved'

# [修正3] 修改变量名，防止覆盖上面的 DATA_DIR 字典
SAVED_DATA_DIR = os.path.join(SAVED_ROOT, 'data') 
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows (保留备用) ##############
DATA_DIR_Win = {
    'MER2023': 'H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\dataset-process',
}

PATH_TO_RAW_FACE_Win = {
    'MER2023': os.path.join(DATA_DIR_Win['MER2023'], 'video'),
}

PATH_TO_FEATURES_Win = {
    'MER2023': os.path.join(DATA_DIR_Win['MER2023'], 'features'),
}

PATH_TO_OPENFACE_Win = "H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\tools\\openface_win_x64"