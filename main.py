"""
Main program for manufacturing data analysis
主程序：训练所有模型，评估性能，生成可视化
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_loader import DataLoader
from utils.trainer import ModelTrainer
from utils.evaluator import ModelEvaluator

# Import all models from separate files
from models.proposed_model import build_proposed_model
from models.lstm_model import build_lstm_model
from models.gru_model import build_gru_model
from models.bilstm_model import build_bilstm_model
from models.cnn_lstm_model import build_cnn_lstm_model
from models.attention_lstm_model import build_attention_lstm_model
from models.cnn_model import build_cnn_model
from models.transformer_model import build_transformer_model
from models.mlp_model import build_mlp_model

# Import visualizers
from visualizations.prediction_viz import PredictionVisualizer
from visualizations.roc_viz import ROCVisualizer
from visualizations.gradcam_viz import GradCAMVisualizer
from visualizations.advanced_viz import AdvancedVisualizer


def create_directories():
    """Create necessary directories for results"""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.TABLE_DIR, exist_ok=True)
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    print(f"✓ Created directories:")
    print(f"  - Models: {config.MODELS_DIR}")
    print(f"  - Tables: {config.TABLE_DIR}")
    print(f"  - Figures: {config.FIGURE_DIR}\n")


def load_and_prepare_data():
    """Load and prepare the manufacturing dataset"""
    print("="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80)

    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_data()

    print(f"\n✓ Data preparation completed!")
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def build_all_models(input_shape):
    """Build all models for comparison"""
    print("="*80)
    print("STEP 2: BUILDING ALL MODELS")
    print("="*80)

    models = {}

    # Define model builders with their names matching config.MODELS_TO_TRAIN
    model_builders = {
        'Proposed': build_proposed_model,
        'LSTM': build_lstm_model,
        'GRU': build_gru_model,
        'BiLSTM': build_bilstm_model,
        'CNN-LSTM': build_cnn_lstm_model,
        'Attention-LSTM': build_attention_lstm_model,
        '1D-CNN': build_cnn_model,
        'Transformer': build_transformer_model,
        'MLP': build_mlp_model,
    }

    # Build only models specified in config
    for idx, model_name in enumerate(config.MODELS_TO_TRAIN, 1):
        if model_name in model_builders:
            print(f"{idx}. Building {model_name}...")
            models[model_name] = model_builders[model_name](input_shape)
        else:
            print(f"Warning: {model_name} not found in model builders")

    print(f"\n✓ {len(models)} models built successfully!\n")

    return models


def train_all_models(models, X_train, y_train, X_val, y_val):
    """Train all models"""
    print("="*80)
    print("STEP 3: TRAINING ALL MODELS")
    print("="*80)

    trainers = {}
    histories = {}

    for idx, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n[{idx}/{len(models)}] Training {model_name}...")
        print("-"*80)

        trainer = ModelTrainer(model, model_name)
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=1
        )

        trainers[model_name] = trainer
        histories[model_name] = history

        # Save final model
        trainer.save_final_model()

    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80 + "\n")

    return trainers, histories


def evaluate_all_models(trainers, X_test, y_test):
    """Evaluate all models on test set"""
    print("="*80)
    print("STEP 4: EVALUATING ALL MODELS")
    print("="*80)

    evaluator = ModelEvaluator()
    all_predictions = {}

    for idx, (model_name, trainer) in enumerate(trainers.items(), 1):
        print(f"\n[{idx}/{len(trainers)}] Evaluating {model_name}...")

        # Load best model
        trainer.load_best_model()

        # Get predictions
        y_pred_prob = trainer.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_pred_prob = y_pred_prob.flatten()

        # Calculate metrics
        evaluator.add_model_results(model_name, y_test, y_pred, y_pred_prob)

        # Store predictions
        all_predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }

    # Print and save comparison table
    print("\n" + "="*80)
    evaluator.print_comparison_table()

    # Save to CSV
    table_path = os.path.join(config.TABLE_DIR, 'performance_comparison.csv')
    evaluator.save_comparison_table(table_path)

    # Get best model
    best_model, best_auc = evaluator.get_best_model('AUC')
    print(f"\n✓ Best Model: {best_model} (AUC = {best_auc:.4f})")
    print("="*80 + "\n")

    return evaluator, all_predictions


def create_all_visualizations(evaluator, all_predictions, histories,
                               trainers, X_test, y_test):
    """Create all visualizations"""
    print("="*80)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*80)

    # 1. Prediction results visualization (3x5 grid)
    print("\n1. Creating prediction results visualization (3x5 grid)...")
    pred_viz = PredictionVisualizer()
    for model_name, preds in all_predictions.items():
        pred_viz.add_model_predictions(
            model_name, y_test, preds['y_pred'], preds['y_pred_prob']
        )

    pred_viz.plot_predictions_grid(
        n_rows=config.PRED_GRID_ROWS,
        n_cols=config.PRED_GRID_COLS,
        save_path=os.path.join(config.FIGURE_DIR, 'predictions_grid.png')
    )
    pred_viz.plot_probability_distribution(
        n_rows=config.PRED_GRID_ROWS,
        n_cols=config.PRED_GRID_COLS,
        save_path=os.path.join(config.FIGURE_DIR, 'probability_distribution.png')
    )
    pred_viz.plot_prediction_errors(
        n_rows=config.PRED_GRID_ROWS,
        n_cols=config.PRED_GRID_COLS,
        save_path=os.path.join(config.FIGURE_DIR, 'prediction_errors.png')
    )

    # 2. ROC curves (1x3 layout)
    print("2. Creating ROC curves (1x3 layout)...")
    roc_viz = ROCVisualizer()
    for model_name, preds in all_predictions.items():
        roc_viz.add_model_results(model_name, y_test, preds['y_pred_prob'])

    # Group models for 1x3 layout
    model_names = list(all_predictions.keys())
    n = len(model_names)
    group_size = (n + 2) // 3
    groups = [
        model_names[i:i+group_size]
        for i in range(0, n, group_size)
    ]
    while len(groups) < 3:
        groups.append([])

    roc_viz.plot_roc_curves_grouped(
        groups=groups,
        save_path=os.path.join(config.FIGURE_DIR, 'roc_curves_grouped.png')
    )
    roc_viz.plot_roc_curves_all(
        save_path=os.path.join(config.FIGURE_DIR, 'roc_curves_all.png')
    )

    # 3. Advanced visualizations
    print("3. Creating advanced visualizations...")
    adv_viz = AdvancedVisualizer()
    for model_name, preds in all_predictions.items():
        history = histories.get(model_name)
        adv_viz.add_model_data(model_name, y_test, preds['y_pred'], history)

    # Confusion matrices
    adv_viz.plot_confusion_matrices(
        n_rows=3, n_cols=3,
        save_path=os.path.join(config.FIGURE_DIR, 'confusion_matrices.png')
    )

    # Learning curves
    print("4. Creating learning curves...")
    adv_viz.plot_learning_curves(
        n_rows=3, n_cols=3,
        save_path=os.path.join(config.FIGURE_DIR, 'learning_curves.png')
    )
    adv_viz.plot_accuracy_curves(
        n_rows=3, n_cols=3,
        save_path=os.path.join(config.FIGURE_DIR, 'accuracy_curves.png')
    )

    # Metrics comparison bar chart
    print("5. Creating metrics comparison bar chart...")
    metrics_dict = {}
    for model_name in all_predictions.keys():
        metrics = evaluator.results[model_name]
        metrics_dict[model_name] = {
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'AUC': metrics['AUC'],
            'F1-Score': metrics['F1-Score']
        }
    adv_viz.plot_metrics_comparison_bar(
        metrics_dict,
        save_path=os.path.join(config.FIGURE_DIR, 'metrics_comparison.png')
    )

    # 6. Grad-CAM heatmaps (3x5 grid)
    print("6. Creating Grad-CAM heatmaps (3x5 grid)...")
    gradcam_viz = GradCAMVisualizer()
    sample_size = min(100, len(X_test))
    X_samples = X_test[:sample_size]

    # Try to generate Grad-CAM for compatible models
    for model_name, trainer in list(trainers.items()):
        try:
            gradcam_viz.add_model_heatmap(model_name, trainer.model, X_samples)
        except Exception as e:
            print(f"   Warning: Could not generate Grad-CAM for {model_name}: {e}")

    if gradcam_viz.heatmaps:
        gradcam_viz.plot_heatmaps_grid(
            n_rows=config.GRADCAM_GRID_ROWS,
            n_cols=config.GRADCAM_GRID_COLS,
            save_path=os.path.join(config.FIGURE_DIR, 'gradcam_heatmaps.png')
        )

    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print(f"  Results saved to: {config.FIGURE_DIR}")
    print("="*80 + "\n")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("MANUFACTURING DATA ANALYSIS - DEEP LEARNING MODEL COMPARISON")
    print("制造业数据分析 - 深度学习模型对比")
    print("="*80 + "\n")

    # Create directories
    create_directories()

    # Step 1: Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_prepare_data()

    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Step 2: Build all models
    models = build_all_models(input_shape)

    # Step 3: Train all models
    trainers, histories = train_all_models(models, X_train, y_train, X_val, y_val)

    # Step 4: Evaluate all models
    evaluator, all_predictions = evaluate_all_models(trainers, X_test, y_test)

    # Step 5: Create all visualizations
    create_all_visualizations(
        evaluator, all_predictions, histories, trainers, X_test, y_test
    )

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nResults Summary:")
    print(f"  📊 Performance table: {config.TABLE_DIR}/performance_comparison.csv")
    print(f"  🤖 Saved models: {config.MODELS_DIR}/")
    print(f"  📈 Visualizations: {config.FIGURE_DIR}/")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
