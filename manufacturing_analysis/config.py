"""
Configuration file for manufacturing data analysis project
"""

import os

# Paths - 请根据您的实际路径修改
BASE_DIR = "/Users/yoyo4477/Documents/数学建模/单子/001EOEG/code"
DATA_PATH = os.path.join(BASE_DIR, "Manufacturing_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data parameters
FEATURES = ['Temperature', 'Machine_Performance', 'Production_Quality_Score',
            'Vibration_Level', 'Energy_Consumption']
TARGET = 'Optimal_Condition'
SEQUENCE_LENGTH = 10  # 时间序列长度

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
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
