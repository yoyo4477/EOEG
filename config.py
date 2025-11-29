"""
Manufacturing Data Analysis - Configuration File
制造业数据分析 - 配置文件
"""

import os

# ==================== 路径配置 ====================
BASE_DIR = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code"

# 数据集路径（选择其中一个）
DATA_PATH = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/data/Manufacturing_dataset.csv"
# 如果数据集在code文件夹，取消下面的注释并注释掉上面的
# DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TABLE_DIR = os.path.join(OUTPUT_DIR, "table")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figure")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# ==================== 数据参数 ====================
FEATURES = ['Temperature', 'Machine_Performance', 'Production_Quality_Score',
            'Vibration_Level', 'Energy_Consumption']
TARGET = 'Optimal_Condition'
SEQUENCE_LENGTH = 10

# 数据划分：训练集:验证集:测试集 = 8:1:1
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_STATE = 42

# ==================== 训练参数 ====================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5

# ==================== 模型参数 ====================
LSTM_UNITS = [128, 64]
GRU_UNITS = [128, 64]
BILSTM_UNITS = [128, 64]
CNN_FILTERS = [64, 128]
CNN_KERNEL_SIZE = 3
TRANSFORMER_HEADS = 4
TRANSFORMER_DIM = 128
DENSE_UNITS = [64, 32]
DROPOUT_RATE = 0.3

# ==================== 可视化参数 ====================
FIGURE_DPI = 300
FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 16           # 图例
AXIS_LABELSIZE = 18      # 坐标轴（大）
TICK_LABELSIZE = 14      # 刻度
TITLE_FONTSIZE = 20

# 图表布局
PRED_GRID_ROWS = 3
PRED_GRID_COLS = 5
GRADCAM_GRID_ROWS = 3
GRADCAM_GRID_COLS = 5
ROC_GRID_COLS = 3

# 子图大小
SUBPLOT_WIDTH = 4
SUBPLOT_HEIGHT = 3

# ==================== 要训练的模型 ====================
MODELS_TO_TRAIN = [
    'Proposed',      # 论文主模型
    'LSTM',
    'GRU',
    'BiLSTM',
    'CNN-LSTM',
    'Attention-LSTM',
    '1D-CNN',
    'Transformer',
    'MLP',
]

# ==================== 系统设置 ====================
import numpy as np
import random

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass
