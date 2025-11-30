"""
Unified model training framework with best model checkpointing
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config


class ModelTrainer:
    """Train and evaluate models with automatic checkpointing"""

    def __init__(self, model, model_name, models_dir=config.MODELS_DIR):
        self.model = model
        self.model_name = model_name
        self.models_dir = models_dir
        self.history = None

        # Create models directory if not exists
        os.makedirs(self.models_dir, exist_ok=True)

    def get_callbacks(self):
        """Get training callbacks"""
        model_path = os.path.join(self.models_dir, f'{self.model_name}_best.h5')

        callbacks = [
            # Save best model after each epoch
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=config.EPOCHS,
              batch_size=config.BATCH_SIZE, verbose=1):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity mode

        Returns:
            Training history
        """
        print(f"\nTraining {self.model_name}...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=verbose
        )

        print(f"\n{self.model_name} training completed!")
        print(f"Best validation loss: {min(self.history.history['val_loss']):.4f}")

        return self.history

    def load_best_model(self):
        """Load the best saved model"""
        model_path = os.path.join(self.models_dir, f'{self.model_name}_best.h5')

        if os.path.exists(model_path):
            print(f"Loading best model from {model_path}")

            # For Proposed model, we need to provide custom objects
            if self.model_name == 'Proposed':
                from models.proposed_model import FeatureEngineeringLayer, ConstraintAwareLoss
                custom_objects = {
                    'FeatureEngineeringLayer': FeatureEngineeringLayer,
                    'ConstraintAwareLoss': ConstraintAwareLoss
                }
                self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
            else:
                self.model = keras.models.load_model(model_path, compile=False)
                # Recompile with original settings
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                    loss='binary_crossentropy',
                    metrics=['accuracy', keras.metrics.Precision(),
                             keras.metrics.Recall(), keras.metrics.AUC(name='auc')]
                )
            return self.model
        else:
            print(f"No saved model found at {model_path}")
            return None

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {self.model_name}...")

        # Get predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Calculate metrics
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))

        # Add predictions for further analysis
        metrics['y_pred'] = y_pred.flatten()
        metrics['y_pred_prob'] = y_pred_prob.flatten()
        metrics['y_true'] = y_test

        return metrics

    def save_final_model(self):
        """Save the final trained model"""
        final_path = os.path.join(self.models_dir, f'{self.model_name}_final.h5')
        self.model.save(final_path)
        print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    # Test trainer
    from utils.data_loader import DataLoader
    from models.lstm_model import build_lstm_model

    # Load data
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_data()

    # Build model
    model = build_lstm_model((config.SEQUENCE_LENGTH, len(config.FEATURES)))

    # Train
    trainer = ModelTrainer(model, "LSTM_Test")
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=5)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    print(f"\nTest metrics: {metrics}")
