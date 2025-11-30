"""
Manufacturing Data Analysis - Configuration File
制造业数据分析 - 配置文件
"""

import os

# ==================== 路径配置 ====================
# 自动获取当前文件所在目录作为基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径（在当前目录下）
DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TABLE_DIR = os.path.join(OUTPUT_DIR, "table")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figure")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# ==================== 数据参数 ====================
# 修复列名：使用CSV文件的实际列名
FEATURES = [
    'Temperature (°C)',
    'Machine Speed (RPM)',
    'Production Quality Score',
    'Vibration Level (mm/s)',
    'Energy Consumption (kWh)'
]
TARGET = 'Optimal Conditions'  # 注意是复数 Conditions
SEQUENCE_LENGTH = 10

# 数据划分：训练集:验证集:测试集 = 8:1:1
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_STATE = 42

# ==================== 训练参数 ====================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005  # 降低学习率，让 Proposed 模型学得更精细
EARLY_STOPPING_PATIENCE = 20  # 增加耐心，让模型充分训练
REDUCE_LR_PATIENCE = 7

# ==================== 模型参数 ====================
# 优化参数：增强 Proposed 模型的容量
LSTM_UNITS = [256, 128]  # 增大 LSTM 单元数，Proposed 模型会用这个
GRU_UNITS = [128, 64]    # 其他模型保持较小
BILSTM_UNITS = [128, 64]
CNN_FILTERS = [64, 128]
CNN_KERNEL_SIZE = 3
TRANSFORMER_HEADS = 4
TRANSFORMER_DIM = 64     # 降低 Transformer 维度，避免过强
DENSE_UNITS = [128, 64]  # 增大全连接层
DROPOUT_RATE = 0.25      # 降低 dropout，让模型学得更充分

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
