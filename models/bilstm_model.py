"""
Bidirectional LSTM Model for Manufacturing Quality Prediction
双向LSTM模型用于制造业质量预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_bilstm_model(input_shape):
    """
    Build Bidirectional LSTM model for binary classification
    构建双向LSTM二分类模型

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First Bidirectional LSTM layer
        layers.Bidirectional(
            layers.LSTM(
                config.BILSTM_UNITS[0],
                return_sequences=True
            ),
            name='bilstm_1'
        ),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),

        # Second Bidirectional LSTM layer
        layers.Bidirectional(
            layers.LSTM(
                config.BILSTM_UNITS[1],
                return_sequences=False
            ),
            name='bilstm_2'
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
    ], name='BiLSTM')

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
    model = build_bilstm_model(input_shape)
    model.summary()

    print(f"\n✓ BiLSTM Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
