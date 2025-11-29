"""
Transformer Model for Manufacturing Quality Prediction
Transformer模型用于制造业质量预测
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            name='multi_head_attention'
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='relu'),
            layers.Dense(self.embed_dim),
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config


def build_transformer_model(input_shape):
    """
    Build Transformer model for binary classification
    构建Transformer二分类模型

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Project input features to embedding dimension
    x = layers.Dense(config.TRANSFORMER_DIM, name='feature_projection')(inputs)

    # Positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embeddings = layers.Embedding(
        input_dim=input_shape[0],
        output_dim=config.TRANSFORMER_DIM,
        name='position_embedding'
    )(positions)
    x = x + position_embeddings

    # Transformer blocks
    x = TransformerBlock(
        embed_dim=config.TRANSFORMER_DIM,
        num_heads=config.TRANSFORMER_HEADS,
        ff_dim=config.TRANSFORMER_DIM * 2,
        rate=config.DROPOUT_RATE,
        name='transformer_block_1'
    )(x)

    x = TransformerBlock(
        embed_dim=config.TRANSFORMER_DIM,
        num_heads=config.TRANSFORMER_HEADS,
        ff_dim=config.TRANSFORMER_DIM * 2,
        rate=config.DROPOUT_RATE,
        name='transformer_block_2'
    )(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Dense layers
    x = layers.Dense(config.DENSE_UNITS[0], activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.Dense(config.DENSE_UNITS[1], activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='Transformer')

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
    model = build_transformer_model(input_shape)
    model.summary()

    print(f"\n✓ Transformer Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
