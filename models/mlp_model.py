"""
MLP Model for Manufacturing Quality Prediction
多层感知机模型用于制造业质量预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_mlp_model(input_shape):
    """
    Build MLP model for binary classification
    构建MLP二分类模型 (先展平序列)

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # Flatten sequence
        layers.Flatten(),

        # First Dense layer
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),

        # Second Dense layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),

        # Third Dense layer
        layers.Dense(config.DENSE_UNITS[0], activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),

        # Fourth Dense layer
        layers.Dense(config.DENSE_UNITS[1], activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),

        # Output layer
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='MLP')

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
    model = build_mlp_model(input_shape)
    model.summary()

    print(f"\n✓ MLP Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
