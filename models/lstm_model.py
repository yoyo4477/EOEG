"""
LSTM Model for Manufacturing Quality Prediction
LSTM模型用于制造业质量预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_lstm_model(input_shape):
    """
    Build LSTM model for binary classification
    构建LSTM二分类模型

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First LSTM layer
        layers.LSTM(
            config.LSTM_UNITS[0],
            return_sequences=True,
            name='lstm_1'
        ),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),

        # Second LSTM layer
        layers.LSTM(
            config.LSTM_UNITS[1],
            return_sequences=False,
            name='lstm_2'
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
    ], name='LSTM')

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
    model = build_lstm_model(input_shape)
    model.summary()

    print(f"\n✓ LSTM Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
