"""
GRU Model for Manufacturing Quality Prediction
GRU模型用于制造业质量预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_gru_model(input_shape):
    """
    Build GRU model for binary classification
    构建GRU二分类模型

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First GRU layer
        layers.GRU(
            config.GRU_UNITS[0],
            return_sequences=True,
            name='gru_1'
        ),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),

        # Second GRU layer
        layers.GRU(
            config.GRU_UNITS[1],
            return_sequences=False,
            name='gru_2'
        ),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),

        # Dense layers
        layers.Dense(config.DENSE_UNITS[0], activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),

        layers.Dense(config.DENSE_UNITS[1], activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),

        # Output layer
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='GRU')

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model


if __name__ == "__main__":
    # Test model
    input_shape = (config.SEQUENCE_LENGTH, len(config.FEATURES))
    model = build_gru_model(input_shape)
    model.summary()

    print(f"\n✓ GRU Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
