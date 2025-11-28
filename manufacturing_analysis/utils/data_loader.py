"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


class DataLoader:
    """Load and preprocess manufacturing dataset"""

    def __init__(self, data_path=config.DATA_PATH):
        self.data_path = data_path
        self.scaler = StandardScaler()

    def load_data(self):
        """Load CSV data"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # Print dataset info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())

        return df

    def preprocess_data(self, df):
        """Preprocess data for model training"""
        # Map Chinese column names to English
        column_mapping = {
            '时间': 'Time',
            '温度': 'Temperature',
            '机器执行': 'Machine_Performance',
            '生产质量评分': 'Production_Quality_Score',
            '振动水平': 'Vibration_Level',
            '能源消耗量': 'Energy_Consumption',
            '最佳条件': 'Optimal_Condition'
        }

        # Check if columns are in Chinese and rename
        if '时间' in df.columns:
            df = df.rename(columns=column_mapping)

        # Extract features and target
        X = df[config.FEATURES].values
        y = df[config.TARGET].values

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def create_sequences(self, X, y):
        """Create sequences for time series models"""
        X_seq, y_seq = [], []

        for i in range(len(X) - config.SEQUENCE_LENGTH):
            X_seq.append(X[i:i + config.SEQUENCE_LENGTH])
            y_seq.append(y[i + config.SEQUENCE_LENGTH])

        return np.array(X_seq), np.array(y_seq)

    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.TEST_SPLIT,
            random_state=config.RANDOM_STATE, stratify=y
        )

        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT),
            random_state=config.RANDOM_STATE, stratify=y_temp
        )

        print(f"\nData split:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_data(self):
        """Complete data preparation pipeline"""
        # Load data
        df = self.load_data()

        # Preprocess
        X, y = self.preprocess_data(df)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_seq, y_seq)

        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_data()
    print("\nData preparation completed successfully!")
