"""
Advanced visualizations: confusion matrix, feature importance, learning curves, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import config

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = config.FONT_FAMILY
matplotlib.rcParams['font.size'] = config.FONT_SIZE


class AdvancedVisualizer:
    """Create advanced visualizations for model analysis"""

    def __init__(self):
        self.data = {}

    def add_model_data(self, model_name, y_true, y_pred, history=None):
        """
        Add model data for visualization

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            history: Training history object
        """
        self.data[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'history': history
        }

    def plot_confusion_matrices(self, n_rows=3, n_cols=3, save_path=None):
        """
        Plot confusion matrices for all models in a grid

        Args:
            n_rows: Number of rows
            n_cols: Number of columns
            save_path: Path to save the figure
        """
        n_models = len(self.data)
        total_subplots = n_rows * n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten()

        for idx, (model_name, data) in enumerate(self.data.items()):
            if idx >= total_subplots:
                break

            ax = axes[idx]
            y_true = data['y_true']
            y_pred = data['y_pred']

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        cbar_kws={'label': 'Count'},
                        annot_kws={'size': config.FONT_SIZE})

            ax.set_title(model_name, fontsize=config.FONT_SIZE + 2, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=config.AXIS_LABELSIZE,
                          fontweight='bold')
            ax.set_ylabel('True Label', fontsize=config.AXIS_LABELSIZE,
                          fontweight='bold')
            ax.tick_params(labelsize=config.TICK_LABELSIZE)

        # Hide unused subplots
        for idx in range(n_models, total_subplots):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")

        return fig

    def plot_learning_curves(self, n_rows=3, n_cols=3, save_path=None):
        """
        Plot learning curves (loss and accuracy) for all models

        Args:
            n_rows: Number of rows
            n_cols: Number of columns
            save_path: Path to save the figure
        """
        # Count models with history
        models_with_history = {
            name: data for name, data in self.data.items()
            if data.get('history') is not None
        }

        if not models_with_history:
            print("No training history available")
            return None

        n_models = len(models_with_history)
        total_subplots = n_rows * n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten()

        for idx, (model_name, data) in enumerate(models_with_history.items()):
            if idx >= total_subplots:
                break

            ax = axes[idx]
            history = data['history']

            # Plot loss
            epochs = range(1, len(history.history['loss']) + 1)
            ax.plot(epochs, history.history['loss'], 'b-', linewidth=2,
                    label='Training Loss')
            ax.plot(epochs, history.history['val_loss'], 'r-', linewidth=2,
                    label='Validation Loss')

            ax.set_title(model_name, fontsize=config.FONT_SIZE + 2, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=config.AXIS_LABELSIZE, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=config.AXIS_LABELSIZE, fontweight='bold')
            ax.legend(loc='best', fontsize=config.TICK_LABELSIZE + 2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=config.TICK_LABELSIZE)

        # Hide unused subplots
        for idx in range(n_models, total_subplots):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Learning curves saved to {save_path}")

        return fig

    def plot_accuracy_curves(self, n_rows=3, n_cols=3, save_path=None):
        """
        Plot accuracy curves for all models

        Args:
            n_rows: Number of rows
            n_cols: Number of columns
            save_path: Path to save the figure
        """
        # Count models with history
        models_with_history = {
            name: data for name, data in self.data.items()
            if data.get('history') is not None
        }

        if not models_with_history:
            print("No training history available")
            return None

        n_models = len(models_with_history)
        total_subplots = n_rows * n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten()

        for idx, (model_name, data) in enumerate(models_with_history.items()):
            if idx >= total_subplots:
                break

            ax = axes[idx]
            history = data['history']

            # Plot accuracy
            if 'accuracy' in history.history:
                epochs = range(1, len(history.history['accuracy']) + 1)
                ax.plot(epochs, history.history['accuracy'], 'b-', linewidth=2,
                        label='Training Accuracy')
                ax.plot(epochs, history.history['val_accuracy'], 'r-', linewidth=2,
                        label='Validation Accuracy')

                ax.set_title(model_name, fontsize=config.FONT_SIZE + 2,
                             fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=config.AXIS_LABELSIZE,
                              fontweight='bold')
                ax.set_ylabel('Accuracy', fontsize=config.AXIS_LABELSIZE,
                              fontweight='bold')
                ax.legend(loc='best', fontsize=config.TICK_LABELSIZE + 2)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=config.TICK_LABELSIZE)

        # Hide unused subplots
        for idx in range(n_models, total_subplots):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Accuracy curves saved to {save_path}")

        return fig

    def plot_metrics_comparison_bar(self, metrics_dict, save_path=None):
        """
        Plot bar chart comparing metrics across models

        Args:
            metrics_dict: Dictionary of {model_name: {metric_name: value}}
            save_path: Path to save the figure
        """
        if not metrics_dict:
            print("No metrics provided")
            return None

        # Extract metrics
        metric_names = list(next(iter(metrics_dict.values())).keys())
        model_names = list(metrics_dict.keys())

        # Filter numeric metrics only
        numeric_metrics = []
        for metric in metric_names:
            try:
                float(metrics_dict[model_names[0]][metric])
                numeric_metrics.append(metric)
            except:
                continue

        n_metrics = len(numeric_metrics)
        if n_metrics == 0:
            print("No numeric metrics found")
            return None

        # Create subplots
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for idx, metric in enumerate(numeric_metrics):
            if idx >= n_rows * n_cols:
                break

            ax = axes[idx]
            values = [float(metrics_dict[model][metric]) for model in model_names]

            # Create bar plot
            bars = ax.bar(range(len(model_names)), values, color='skyblue',
                           edgecolor='navy', linewidth=1.5)

            # Highlight best model
            if metric in ['MAE', 'RMSE', 'MAPE']:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')

            ax.set_title(metric, fontsize=config.FONT_SIZE + 2, fontweight='bold')
            ax.set_xlabel('Model', fontsize=config.AXIS_LABELSIZE)
            ax.set_ylabel(metric, fontsize=config.AXIS_LABELSIZE)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right',
                               fontsize=config.TICK_LABELSIZE)
            ax.tick_params(labelsize=config.TICK_LABELSIZE)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}',
                        ha='center', va='bottom', fontsize=config.TICK_LABELSIZE)

        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Test advanced visualizations
    np.random.seed(42)

    # Create visualizer
    viz = AdvancedVisualizer()

    # Add simulated data for models
    model_names = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'Attention-LSTM',
                   'CNN', 'Transformer', 'MLP', 'Proposed Model']

    for model_name in model_names:
        n_samples = 500
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        viz.add_model_data(model_name, y_true, y_pred)

    # Create confusion matrices
    viz.plot_confusion_matrices(save_path='/tmp/confusion_matrices.png')

    # Create metrics comparison
    metrics_dict = {
        model: {
            'Accuracy': np.random.rand(),
            'Precision': np.random.rand(),
            'Recall': np.random.rand(),
            'F1-Score': np.random.rand(),
        }
        for model in model_names
    }
    viz.plot_metrics_comparison_bar(metrics_dict, save_path='/tmp/metrics_comparison.png')

    print("Advanced visualizations created successfully!")
