"""
Proposed Model - Three-Module Architecture
论文主模型 - 三模块架构：Baseline + Feature Engineering + Constraint-Aware Loss
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config


class FeatureEngineeringLayer(layers.Layer):
    """Module A: Enhanced Feature Engineering Layer - 增强版特征工程"""

    def __init__(self, enhanced_dim=64, **kwargs):  # 增加到64维
        super().__init__(**kwargs)
        self.enhanced_dim = enhanced_dim

    def build(self, input_shape):
        # 多层特征提取网络
        self.dense1 = layers.Dense(self.enhanced_dim, activation='relu')
        self.batch_norm1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(self.enhanced_dim, activation='relu')
        self.batch_norm2 = layers.BatchNormalization()
        self.dense3 = layers.Dense(self.enhanced_dim // 2, activation='relu')  # 32维
        self.batch_norm3 = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, inputs):
        # 深层特征提取
        x = self.dense1(inputs)
        x = self.batch_norm1(x)
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.dense3(x)
        x = self.batch_norm3(x)

        # Concatenate original features with engineered features
        enhanced = tf.concat([inputs, x], axis=-1)
        return enhanced

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.enhanced_dim // 2,)

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
    Proposed Three-Module Model - Enhanced Version
    提案的三模块模型 - 增强版

    优化策略：
    1. 更深的 LSTM 网络（3层）
    2. 更强的特征工程（64维）
    3. 更大的模型容量
    4. 更强的约束损失

    Architecture:
    1. Baseline: Deep LSTM Multi-Task Network (3层)
    2. Module A: Enhanced Feature Engineering Layer (64维)
    3. Module B: Stronger Constraint-Aware Loss Function
    """

    inputs = layers.Input(shape=input_shape, name='input')

    # Module A: Enhanced Feature Engineering (64维特征增强)
    x = layers.TimeDistributed(
        FeatureEngineeringLayer(enhanced_dim=64),  # 增加到64维
        name='feature_engineering_module'
    )(inputs)

    # Baseline: 3-Layer Deep LSTM Network (更深的网络)
    # Layer 1: 256 units
    x = layers.LSTM(
        config.LSTM_UNITS[0],  # 256
        return_sequences=True,
        name='lstm_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Layer 2: 128 units
    x = layers.LSTM(
        config.LSTM_UNITS[1],  # 128
        return_sequences=True,  # 保持序列，再加一层
        name='lstm_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Layer 3: 64 units (新增第三层LSTM)
    x = layers.LSTM(
        64,
        return_sequences=False,
        name='lstm_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Dense layers (更深的全连接层)
    # Layer 1: 128 units
    x = layers.Dense(config.DENSE_UNITS[0], activation='relu')(x)  # 128
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Layer 2: 64 units
    x = layers.Dense(config.DENSE_UNITS[1], activation='relu')(x)  # 64
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Layer 3: 32 units (新增一层)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Proposed')

    # Compile with Module B: Enhanced Constraint-Aware Loss (更强的约束)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=ConstraintAwareLoss(alpha=0.15, beta=0.08),  # 增加惩罚权重
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
