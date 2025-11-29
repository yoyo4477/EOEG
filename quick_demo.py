"""
Quick demo - Train 3 key models and show results
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_loader import DataLoader
from utils.trainer import ModelTrainer
from utils.evaluator import ModelEvaluator

# Import key models
from models.proposed_model import build_proposed_model
from models.lstm_model import build_lstm_model
from models.transformer_model import build_transformer_model

# Import visualizers
from visualizations.prediction_viz import PredictionVisualizer
from visualizations.roc_viz import ROCVisualizer
from visualizations.advanced_viz import AdvancedVisualizer

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("\n" + "="*80)
print("QUICK DEMO - Manufacturing Data Analysis")
print("Training 3 key models for demonstration")
print("="*80 + "\n")

# Create directories
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# Step 1: Load data
print("STEP 1: Loading data...")
print("-"*80)
loader = DataLoader()
X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_data()

input_shape = (X_train.shape[1], X_train.shape[2])
print(f"Input shape: {input_shape}\n")

# Step 2: Build models
print("STEP 2: Building models...")
print("-"*80)
models = {}

print("1. Building Proposed Model (Ours)...")
models['Proposed Model'] = build_proposed_model(input_shape)

print("2. Building LSTM...")
models['LSTM'] = build_lstm_model(input_shape)

print("3. Building Transformer...")
models['Transformer'] = build_transformer_model(input_shape)

print(f"\n{len(models)} models built successfully!\n")

# Step 3: Train models
print("STEP 3: Training models...")
print("-"*80)
trainers = {}
histories = {}

for idx, (model_name, model) in enumerate(models.items(), 1):
    print(f"\n[{idx}/{len(models)}] Training {model_name}...")
    print("-"*40)

    trainer = ModelTrainer(model, model_name)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=2  # Less verbose
    )

    trainers[model_name] = trainer
    histories[model_name] = history
    trainer.save_final_model()

print("\n" + "="*80)
print("Training completed!")
print("="*80 + "\n")

# Step 4: Evaluate models
print("STEP 4: Evaluating models...")
print("-"*80)
evaluator = ModelEvaluator()
all_predictions = {}

for idx, (model_name, trainer) in enumerate(trainers.items(), 1):
    print(f"[{idx}/{len(trainers)}] Evaluating {model_name}...")

    trainer.load_best_model()
    y_pred_prob = trainer.model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_pred_prob = y_pred_prob.flatten()

    evaluator.add_model_results(model_name, y_test, y_pred, y_pred_prob)
    all_predictions[model_name] = {'y_pred': y_pred, 'y_pred_prob': y_pred_prob}

# Print comparison table
print("\n" + "="*80)
evaluator.print_comparison_table()

# Save table
table_path = os.path.join(config.RESULTS_DIR, 'performance_comparison.csv')
evaluator.save_comparison_table(table_path)

best_model, best_auc = evaluator.get_best_model('AUC')
print(f"\nBest Model: {best_model} (AUC = {best_auc:.4f})")
print("="*80 + "\n")

# Step 5: Create visualizations
print("STEP 5: Creating visualizations...")
print("-"*80)

# Prediction results
print("1. Creating prediction results...")
pred_viz = PredictionVisualizer()
for model_name, preds in all_predictions.items():
    pred_viz.add_model_predictions(model_name, y_test, preds['y_pred'], preds['y_pred_prob'])

pred_viz.plot_predictions_grid(
    n_rows=1, n_cols=3,
    save_path=os.path.join(config.RESULTS_DIR, 'predictions_grid.png')
)

# ROC curves
print("2. Creating ROC curves...")
roc_viz = ROCVisualizer()
for model_name, preds in all_predictions.items():
    roc_viz.add_model_results(model_name, y_test, preds['y_pred_prob'])

roc_viz.plot_roc_curves_grouped(
    groups=[list(all_predictions.keys())],
    figsize=(8, 6),
    save_path=os.path.join(config.RESULTS_DIR, 'roc_curves.png')
)

# Confusion matrices
print("3. Creating confusion matrices...")
adv_viz = AdvancedVisualizer()
for model_name, preds in all_predictions.items():
    history = histories.get(model_name)
    adv_viz.add_model_data(model_name, y_test, preds['y_pred'], history)

adv_viz.plot_confusion_matrices(
    n_rows=1, n_cols=3,
    save_path=os.path.join(config.RESULTS_DIR, 'confusion_matrices.png')
)

# Learning curves
print("4. Creating learning curves...")
adv_viz.plot_learning_curves(
    n_rows=1, n_cols=3,
    save_path=os.path.join(config.RESULTS_DIR, 'learning_curves.png')
)

# Metrics comparison
print("5. Creating metrics comparison...")
metrics_dict = {}
for model_name in all_predictions.keys():
    metrics = evaluator.results[model_name]
    metrics_dict[model_name] = {
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'AUC': metrics['AUC'],
    }
adv_viz.plot_metrics_comparison_bar(
    metrics_dict,
    save_path=os.path.join(config.RESULTS_DIR, 'metrics_comparison.png')
)

print("\n" + "="*80)
print("DEMO COMPLETED!")
print("="*80)
print("\nResults saved to:")
print(f"  - Performance table: {table_path}")
print(f"  - Models: {config.MODELS_DIR}/")
print(f"  - Visualizations: {config.RESULTS_DIR}/")
print("\n" + "="*80 + "\n")
