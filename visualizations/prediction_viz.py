"""
Prediction results visualization (3 rows x 5 columns layout)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import config

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = config.FONT_FAMILY
matplotlib.rcParams['font.size'] = config.FONT_SIZE


class PredictionVisualizer:
    """Visualize model predictions in a grid layout"""

    def __init__(self, figsize_per_subplot=(4, 3)):
        self.figsize_per_subplot = figsize_per_subplot
        self.results = {}

    def add_model_predictions(self, model_name, y_true, y_pred, y_pred_prob):
        """
        Add model predictions for visualization

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_prob: Predicted probabilities
        """
        self.results[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }

    def plot_predictions_grid(self, n_rows=3, n_cols=5, save_path=None):
        """
        Plot prediction results in a grid layout

        Args:
            n_rows: Number of rows in the grid
            n_cols: Number of columns in the grid
            save_path: Path to save the figure
        """
        n_models = len(self.results)
        total_subplots = n_rows * n_cols

        if n_models > total_subplots:
            print(f"Warning: {n_models} models but only {total_subplots} subplots available")

        # Create figure
        fig_width = n_cols * self.figsize_per_subplot[0]
        fig_height = n_rows * self.figsize_per_subplot[1]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        # Plot each model
        for idx, (model_name, data) in enumerate(self.results.items()):
            if idx >= total_subplots:
                break

            ax = axes[idx]
            y_true = data['y_true']
            y_pred_prob = data['y_pred_prob']

            # Plot true vs predicted probabilities
            samples = min(len(y_true), 200)  # Limit to 200 samples for clarity
            x = np.arange(samples)

            ax.plot(x, y_true[:samples], 'o', label='True', markersize=4, alpha=0.7)
            ax.plot(x, y_pred_prob[:samples], 's', label='Predicted',
                    markersize=3, alpha=0.7)

            ax.set_title(model_name, fontsize=config.FONT_SIZE, fontweight='bold')
            ax.set_xlabel('Sample Index', fontsize=config.AXIS_LABELSIZE)
            ax.set_ylabel('Value', fontsize=config.AXIS_LABELSIZE)
            ax.legend(loc='best', fontsize=config.TICK_LABELSIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=config.TICK_LABELSIZE)

        # Hide unused subplots
        for idx in range(n_models, total_subplots):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Prediction visualization saved to {save_path}")

        return fig

    def plot_probability_distribution(self, n_rows=3, n_cols=5, save_path=None):
        """
        Plot probability distribution for each model

        Args:
            n_rows: Number of rows in the grid
            n_cols: Number of columns in the grid
            save_path: Path to save the figure
        """
        n_models = len(self.results)
        total_subplots = n_rows * n_cols

        # Create figure
        fig_width = n_cols * self.figsize_per_subplot[0]
        fig_height = n_rows * self.figsize_per_subplot[1]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        # Plot each model
        for idx, (model_name, data) in enumerate(self.results.items()):
            if idx >= total_subplots:
                break

            ax = axes[idx]
            y_true = data['y_true']
            y_pred_prob = data['y_pred_prob']

            # Separate probabilities by true class
            prob_class_0 = y_pred_prob[y_true == 0]
            prob_class_1 = y_pred_prob[y_true == 1]

            # Plot histograms
            ax.hist(prob_class_0, bins=30, alpha=0.6, label='Class 0', color='blue')
            ax.hist(prob_class_1, bins=30, alpha=0.6, label='Class 1', color='red')

            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5,
                       label='Threshold')

            ax.set_title(model_name, fontsize=config.FONT_SIZE, fontweight='bold')
            ax.set_xlabel('Predicted Probability', fontsize=config.AXIS_LABELSIZE)
            ax.set_ylabel('Frequency', fontsize=config.AXIS_LABELSIZE)
            ax.legend(loc='best', fontsize=config.TICK_LABELSIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=config.TICK_LABELSIZE)

        # Hide unused subplots
        for idx in range(n_models, total_subplots):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Probability distribution saved to {save_path}")

        return fig

    def plot_prediction_errors(self, n_rows=3, n_cols=5, save_path=None):
        """
        Plot prediction errors for each model

        Args:
            n_rows: Number of rows in the grid
            n_cols: Number of columns in the grid
            save_path: Path to save the figure
        """
        n_models = len(self.results)
        total_subplots = n_rows * n_cols

        # Create figure
        fig_width = n_cols * self.figsize_per_subplot[0]
        fig_height = n_rows * self.figsize_per_subplot[1]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        # Plot each model
        for idx, (model_name, data) in enumerate(self.results.items()):
            if idx >= total_subplots:
                break

            ax = axes[idx]
            y_true = data['y_true']
            y_pred_prob = data['y_pred_prob']

            # Calculate errors
            errors = y_true - y_pred_prob
            samples = min(len(errors), 200)

            ax.plot(errors[:samples], 'o-', markersize=3, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5)

            ax.set_title(model_name, fontsize=config.FONT_SIZE, fontweight='bold')
            ax.set_xlabel('Sample Index', fontsize=config.AXIS_LABELSIZE)
            ax.set_ylabel('Prediction Error', fontsize=config.AXIS_LABELSIZE)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=config.TICK_LABELSIZE)

        # Hide unused subplots
        for idx in range(n_models, total_subplots):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Prediction errors saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Test visualization
    np.random.seed(42)

    # Create visualizer
    viz = PredictionVisualizer()

    # Add simulated results for 9 models
    model_names = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'Attention-LSTM',
                   'CNN', 'Transformer', 'MLP', 'Proposed Model']

    for model_name in model_names:
        n_samples = 300
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_prob = np.random.rand(n_samples)
        y_pred = (y_pred_prob > 0.5).astype(int)
        viz.add_model_predictions(model_name, y_true, y_pred, y_pred_prob)

    # Create visualizations
    viz.plot_predictions_grid(save_path='/tmp/predictions_grid.png')
    viz.plot_probability_distribution(save_path='/tmp/prob_distribution.png')
    viz.plot_prediction_errors(save_path='/tmp/prediction_errors.png')

    print("Visualizations created successfully!")
