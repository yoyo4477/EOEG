"""
Transformer Model for manufacturing quality prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            name='multi_head_attention'
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu', name='ffn_dense_1'),
            layers.Dense(embed_dim, name='ffn_dense_2'),
        ], name='feed_forward')

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_2')
        self.dropout1 = layers.Dropout(rate, name='dropout_1')
        self.dropout2 = layers.Dropout(rate, name='dropout_2')

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config_dict


def build_transformer_model(input_shape):
    """
    Build Transformer model for binary classification

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
        name='transformer_block_1'
    )(x)

    x = TransformerBlock(
        embed_dim=config.TRANSFORMER_DIM,
        num_heads=config.TRANSFORMER_HEADS,
        ff_dim=config.TRANSFORMER_DIM * 2,
        name='transformer_block_2'
    )(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)

    # Dense layers
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='Transformer_Model')

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
    model = build_transformer_model((config.SEQUENCE_LENGTH, len(config.FEATURES)))
    model.summary()
