"""
Configuration file for manufacturing data analysis project
"""

import os

# ==================== 路径配置 ====================
# 基础路径：/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code
BASE_DIR = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code"

# 数据集路径（优先使用code目录下的数据集）
DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
# 如果数据集在data文件夹，使用这个路径：
# DATA_PATH = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/data/Manufacturing_dataset.csv"

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TABLE_DIR = os.path.join(OUTPUT_DIR, "table")    # 表格输出路径
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figure")  # 图片输出路径
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")  # 模型保存路径

# ==================== 数据参数 ====================
FEATURES = ['Temperature', 'Machine_Performance', 'Production_Quality_Score',
            'Vibration_Level', 'Energy_Consumption']
TARGET = 'Optimal_Condition'
SEQUENCE_LENGTH = 10  # 时间序列长度

# ==================== 训练参数 ====================
BATCH_SIZE = 32
EPOCHS = 100  # 完整训练轮数
LEARNING_RATE = 0.001

# 数据划分比例：训练集:验证集:测试集 = 8:1:1
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_STATE = 42

# Model parameters
LSTM_UNITS = [128, 64]
CNN_FILTERS = [64, 128]
TRANSFORMER_HEADS = 4
TRANSFORMER_DIM = 128

# Visualization parameters
FIGURE_DPI = 300
FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 16
AXIS_LABELSIZE = 14
TICK_LABELSIZE = 12
