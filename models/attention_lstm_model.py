"""
Attention-LSTM Model for Manufacturing Quality Prediction
注意力LSTM模型用于制造业质量预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config


class AttentionLayer(layers.Layer):
    """Attention mechanism layer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        super().build(input_shape)

    def call(self, x):
        # Compute attention scores
        e = keras.activations.tanh(tf.matmul(x, self.W) + self.b)
        a = keras.activations.softmax(e, axis=1)
        # Apply attention weights
        output = x * a
        return tf.reduce_sum(output, axis=1)

    def get_config(self):
        return super().get_config()


def build_attention_lstm_model(input_shape):
    """
    Build LSTM with Attention mechanism model for binary classification
    构建带注意力机制的LSTM二分类模型

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # First LSTM layer
    x = layers.LSTM(
        config.LSTM_UNITS[0],
        return_sequences=True,
        name='lstm_1'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Second LSTM layer
    x = layers.LSTM(
        config.LSTM_UNITS[1],
        return_sequences=True,
        name='lstm_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Attention layer
    x = AttentionLayer(name='attention')(x)

    # Dense layers
    x = layers.Dense(config.DENSE_UNITS[0], activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.Dense(config.DENSE_UNITS[1], activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Attention-LSTM')

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
    model = build_attention_lstm_model(input_shape)
    model.summary()

    print(f"\n✓ Attention-LSTM Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
