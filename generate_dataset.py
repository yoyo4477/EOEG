"""
Generate simulated manufacturing dataset
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 2000

# Generate time series data
timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='H')

# Generate features with realistic patterns
data = {
    '时间': timestamps,
    '温度': np.random.normal(75, 10, n_samples) + 5 * np.sin(np.arange(n_samples) / 24),  # Temperature with daily cycle
    '机器执行': np.random.uniform(60, 100, n_samples),  # Machine performance
    '生产质量评分': np.random.normal(85, 8, n_samples),  # Quality score
    '振动水平': np.random.exponential(2, n_samples) + np.random.normal(5, 1, n_samples),  # Vibration
    '能源消耗量': np.random.gamma(2, 20, n_samples) + np.random.normal(50, 5, n_samples),  # Energy consumption
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate target based on features (optimal condition)
# High quality, low vibration, moderate energy -> optimal (1)
quality_score = (df['生产质量评分'] - df['生产质量评分'].mean()) / df['生产质量评分'].std()
vibration_score = -(df['振动水平'] - df['振动水平'].mean()) / df['振动水平'].std()
energy_score = -(df['能源消耗量'] - df['能源消耗量'].mean()) / df['能源消耗量'].std()

# Combined score
combined_score = quality_score + vibration_score + 0.5 * energy_score

# Create binary label with some noise
probabilities = 1 / (1 + np.exp(-combined_score))  # Sigmoid
df['最佳条件'] = (probabilities > 0.5).astype(int)

# Add some noise to make it more realistic
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
df.loc[noise_indices, '最佳条件'] = 1 - df.loc[noise_indices, '最佳条件']

# Save to CSV
output_path = '/home/user/EOEG/Manufacturing_dataset.csv'
df.to_csv(output_path, index=False)

print(f"Generated dataset with {len(df)} samples")
print(f"Saved to: {output_path}")
print(f"\nDataset info:")
print(df.info())
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df['最佳条件'].value_counts())
print(f"\nBasic statistics:")
print(df.describe())
