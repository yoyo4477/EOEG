"""
Data loading and preprocessing utilities
数据加载和预处理工具 - 8:1:1划分
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


class DataLoader:
    """制造业数据集加载和预处理"""

    def __init__(self, data_path=config.DATA_PATH):
        self.data_path = data_path
        self.scaler = StandardScaler()

    def load_data(self):
        """加载CSV数据"""
        print(f"📂 Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)

        print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        print(f"  Columns: {df.columns.tolist()}")

        return df

    def preprocess_data(self, df):
        """数据预处理"""
        # 中英文列名映射 - 映射到config.FEATURES中定义的列名
        column_mapping = {
            '时间': 'Time',
            '温度': 'Temperature (°C)',
            '机器执行': 'Machine Speed (RPM)',
            '生产质量评分': 'Production Quality Score',
            '振动水平': 'Vibration Level (mm/s)',
            '能源消耗量': 'Energy Consumption (kWh)',
            '最佳条件': 'Optimal Conditions'
        }

        # 如果是中文列名，转换为英文
        if '时间' in df.columns:
            df = df.rename(columns=column_mapping)
            print("✓ Column names converted to English")

        # 提取特征和标签
        X = df[config.FEATURES].values
        y = df[config.TARGET].values

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        print(f"✓ Features normalized")
        print(f"  Label distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")

        return X_scaled, y

    def create_sequences(self, X, y):
        """创建时间序列"""
        X_seq, y_seq = [], []

        for i in range(len(X) - config.SEQUENCE_LENGTH):
            X_seq.append(X[i:i + config.SEQUENCE_LENGTH])
            y_seq.append(y[i + config.SEQUENCE_LENGTH])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"✓ Sequences created: {X_seq.shape}")

        return X_seq, y_seq

    def split_data(self, X, y):
        """
        数据划分：训练集:验证集:测试集 = 8:1:1
        """
        # 第一次划分：分离测试集（10%）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_STATE,
            stratify=y
        )

        # 第二次划分：从剩余90%中分离验证集（占总数10%）
        val_ratio = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=config.RANDOM_STATE,
            stratify=y_temp
        )

        # 验证划分比例
        total = len(X)
        train_pct = len(X_train) / total * 100
        val_pct = len(X_val) / total * 100
        test_pct = len(X_test) / total * 100

        print(f"\n📊 Data Split (Train:Val:Test = 8:1:1):")
        print(f"  Train: {len(X_train):5d} samples ({train_pct:.1f}%)")
        print(f"  Val:   {len(X_val):5d} samples ({val_pct:.1f}%)")
        print(f"  Test:  {len(X_test):5d} samples ({test_pct:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_data(self):
        """完整的数据准备流程"""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)

        # 1. 加载数据
        df = self.load_data()

        # 2. 预处理
        X, y = self.preprocess_data(df)

        # 3. 创建序列
        X_seq, y_seq = self.create_sequences(X, y)

        # 4. 划分数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_seq, y_seq)

        print("\n✓ Data preparation complete!")
        print("="*60 + "\n")

        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler


if __name__ == "__main__":
    # 测试数据加载器
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_data()
    print(f"Train shape: {X_train.shape}")
    print(f"Input shape for models: ({X_train.shape[1]}, {X_train.shape[2]})")
