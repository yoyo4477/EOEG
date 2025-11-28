"""
Baseline comparison models for benchmarking
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_gru_model(input_shape):
    """GRU-based model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True, name='gru_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.GRU(64, return_sequences=False, name='gru_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='GRU_Model')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )
    return model


def build_bilstm_model(input_shape):
    """Bidirectional LSTM model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='bilstm_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False), name='bilstm_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='BiLSTM_Model')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )
    return model


class AttentionLayer(layers.Layer):
    """Attention mechanism layer"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Compute attention scores
        e = keras.activations.tanh(tf.matmul(x, self.W) + self.b)
        a = keras.activations.softmax(e, axis=1)
        # Apply attention weights
        output = x * a
        return tf.reduce_sum(output, axis=1)


def build_attention_lstm_model(input_shape):
    """LSTM with Attention mechanism"""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=True, name='lstm_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # Attention layer
    x = AttentionLayer(name='attention')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Attention_LSTM_Model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )
    return model


def build_cnn_model(input_shape):
    """1D CNN model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv1d_1'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', name='conv1d_2'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv1d_3'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='CNN_Model')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )
    return model


def build_mlp_model(input_shape):
    """Simple MLP model (flatten sequence first)"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='MLP_Model')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )
    return model


if __name__ == "__main__":
    # Test all baseline models
    input_shape = (config.SEQUENCE_LENGTH, len(config.FEATURES))

    print("Building GRU model...")
    gru = build_gru_model(input_shape)
    print(f"GRU parameters: {gru.count_params()}\n")

    print("Building BiLSTM model...")
    bilstm = build_bilstm_model(input_shape)
    print(f"BiLSTM parameters: {bilstm.count_params()}\n")

    print("Building Attention-LSTM model...")
    att_lstm = build_attention_lstm_model(input_shape)
    print(f"Attention-LSTM parameters: {att_lstm.count_params()}\n")

    print("Building CNN model...")
    cnn = build_cnn_model(input_shape)
    print(f"CNN parameters: {cnn.count_params()}\n")

    print("Building MLP model...")
    mlp = build_mlp_model(input_shape)
    print(f"MLP parameters: {mlp.count_params()}\n")
