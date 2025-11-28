"""
CNN-LSTM Hybrid Model for manufacturing quality prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_cnn_lstm_model(input_shape):
    """
    Build CNN-LSTM hybrid model for binary classification

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # CNN layers for feature extraction
        layers.Conv1D(
            config.CNN_FILTERS[0],
            kernel_size=3,
            padding='same',
            activation='relu',
            name='conv1d_1'
        ),
        layers.BatchNormalization(name='bn_1'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_1'),
        layers.Dropout(0.2, name='dropout_1'),

        layers.Conv1D(
            config.CNN_FILTERS[1],
            kernel_size=3,
            padding='same',
            activation='relu',
            name='conv1d_2'
        ),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.2, name='dropout_2'),

        # LSTM layers for temporal patterns
        layers.LSTM(
            config.LSTM_UNITS[0],
            return_sequences=True,
            name='lstm_layer_1'
        ),
        layers.BatchNormalization(name='bn_3'),
        layers.Dropout(0.3, name='dropout_3'),

        layers.LSTM(
            config.LSTM_UNITS[1],
            return_sequences=False,
            name='lstm_layer_2'
        ),
        layers.BatchNormalization(name='bn_4'),
        layers.Dropout(0.3, name='dropout_4'),

        # Dense layers
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout_5'),
        layers.Dense(32, activation='relu', name='dense_2'),

        # Output layer
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='CNN_LSTM_Model')

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )

    return model


if __name__ == "__main__":
    # Test model building
    model = build_cnn_lstm_model((config.SEQUENCE_LENGTH, len(config.FEATURES)))
    model.summary()
