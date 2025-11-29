"""
1D CNN Model for Manufacturing Quality Prediction
1D卷积神经网络模型用于制造业质量预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_cnn_model(input_shape):
    """
    Build 1D CNN model for binary classification
    构建1D-CNN二分类模型

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First Conv1D layer
        layers.Conv1D(
            config.CNN_FILTERS[0],
            kernel_size=config.CNN_KERNEL_SIZE,
            padding='same',
            activation='relu',
            name='conv1d_1'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Second Conv1D layer
        layers.Conv1D(
            config.CNN_FILTERS[1],
            kernel_size=config.CNN_KERNEL_SIZE,
            padding='same',
            activation='relu',
            name='conv1d_2'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Third Conv1D layer
        layers.Conv1D(
            config.CNN_FILTERS[0],
            kernel_size=config.CNN_KERNEL_SIZE,
            padding='same',
            activation='relu',
            name='conv1d_3'
        ),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Dense layers
        layers.Dense(config.DENSE_UNITS[0], activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),

        layers.Dense(config.DENSE_UNITS[1], activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),

        # Output layer
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='1D-CNN')

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
    model = build_cnn_model(input_shape)
    model.summary()

    print(f"\n✓ 1D-CNN Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
