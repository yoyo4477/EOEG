"""
Proposed Model - Three-Module Architecture
论文主模型 - 三模块架构：Baseline + Feature Engineering + Constraint-Aware Loss
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config


class FeatureEngineeringLayer(layers.Layer):
    """Module A: Feature Engineering Layer"""

    def __init__(self, enhanced_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.enhanced_dim = enhanced_dim

    def build(self, input_shape):
        self.dense1 = layers.Dense(self.enhanced_dim, activation='relu')
        self.dense2 = layers.Dense(self.enhanced_dim, activation='relu')
        self.batch_norm = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, inputs):
        # Statistical features
        x = self.dense1(inputs)
        x = self.batch_norm(x)
        x = self.dense2(x)

        # Concatenate original features with engineered features
        enhanced = tf.concat([inputs, x], axis=-1)
        return enhanced

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.enhanced_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({"enhanced_dim": self.enhanced_dim})
        return config


class ConstraintAwareLoss(keras.losses.Loss):
    """Module B: Constraint-Aware Penalty Loss"""

    def __init__(self, alpha=0.1, beta=0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # Physical constraint penalty
        self.beta = beta    # Temporal consistency penalty

    def call(self, y_true, y_pred):
        # Reshape to ensure matching dimensions
        y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), tf.float32)

        # Base loss: Binary Cross-Entropy
        bce = keras.losses.binary_crossentropy(y_true, y_pred)

        # Physical constraint penalty: predictions should be in [0, 1]
        physical_penalty = tf.reduce_mean(
            tf.square(tf.maximum(0.0, y_pred - 1.0)) +
            tf.square(tf.maximum(0.0, -y_pred))
        )

        # Temporal consistency penalty: smooth predictions over time
        temporal_penalty = tf.reduce_mean(
            tf.square(y_pred[1:] - y_pred[:-1])
        )

        # Total loss
        total_loss = bce + self.alpha * physical_penalty + self.beta * temporal_penalty

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "beta": self.beta})
        return config


def build_proposed_model(input_shape):
    """
    Proposed Three-Module Model
    提案的三模块模型

    Architecture:
    1. Baseline: LSTM Multi-Task Network
    2. Module A: Feature Engineering Layer
    3. Module B: Constraint-Aware Loss Function
    """

    inputs = layers.Input(shape=input_shape, name='input')

    # Module A: Feature Engineering (applied at each timestep)
    x = layers.TimeDistributed(
        FeatureEngineeringLayer(enhanced_dim=32),
        name='feature_engineering_module'
    )(inputs)

    # Baseline: Multi-Layer LSTM Network
    x = layers.LSTM(
        config.LSTM_UNITS[0],
        return_sequences=True,
        name='lstm_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.LSTM(
        config.LSTM_UNITS[1],
        return_sequences=False,
        name='lstm_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Dense layers
    x = layers.Dense(config.DENSE_UNITS[0], activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.Dense(config.DENSE_UNITS[1], activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Proposed')

    # Compile with Module B: Constraint-Aware Loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=ConstraintAwareLoss(alpha=0.1, beta=0.05),
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
    model = build_proposed_model(input_shape)
    model.summary()

    print(f"\n✓ Proposed Model created successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Total parameters: {model.count_params():,}")
