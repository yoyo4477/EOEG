"""
ROC curve visualization (1 row x 3 columns layout)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc
import config

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = config.FONT_FAMILY
matplotlib.rcParams['font.size'] = config.FONT_SIZE


class ROCVisualizer:
    """Visualize ROC curves for model comparison"""

    def __init__(self):
        self.results = {}

    def add_model_results(self, model_name, y_true, y_pred_prob):
        """
        Add model results for ROC curve

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred_prob: Predicted probabilities
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        self.results[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'y_true': y_true,
            'y_pred_prob': y_pred_prob
        }

    def plot_roc_curves_grouped(self, groups=None, figsize=(18, 5), save_path=None):
        """
        Plot ROC curves in 1 row x 3 columns layout with grouped models

        Args:
            groups: List of 3 lists, each containing model names for that subplot
                   If None, will auto-group models
            figsize: Figure size
            save_path: Path to save the figure
        """
        if groups is None:
            # Auto-group models into 3 groups
            model_names = list(self.results.keys())
            n = len(model_names)
            group_size = (n + 2) // 3
            groups = [
                model_names[i:i+group_size]
                for i in range(0, n, group_size)
            ]
            # Ensure we have exactly 3 groups
            while len(groups) < 3:
                groups.append([])

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Define color palette
        colors = plt.cm.Set3(np.linspace(0, 1, 12))

        for col_idx, (ax, group) in enumerate(zip(axes, groups)):
            # Plot random reference line first
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Reference',
                    alpha=0.7)

            # Plot models in this group
            for model_idx, model_name in enumerate(group):
                if model_name not in self.results:
                    continue

                data = self.results[model_name]
                color = colors[model_idx % len(colors)]

                ax.plot(data['fpr'], data['tpr'],
                        linewidth=2.5,
                        label=f"{model_name} (AUC={data['auc']:.3f})",
                        color=color)

            ax.set_xlabel('False Positive Rate', fontsize=config.AXIS_LABELSIZE,
                          fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=config.AXIS_LABELSIZE,
                          fontweight='bold')
            ax.set_title(f'ROC Curves - Group {col_idx + 1}',
                         fontsize=config.FONT_SIZE + 2, fontweight='bold')
            ax.legend(loc='lower right', fontsize=config.FONT_SIZE,
                      framealpha=0.9, edgecolor='black')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.tick_params(labelsize=config.TICK_LABELSIZE)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")

        return fig

    def plot_roc_curves_all(self, figsize=(12, 10), save_path=None):
        """
        Plot all ROC curves on a single plot

        Args:
            figsize: Figure size
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot random reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5,
                label='Random Reference', alpha=0.8)

        # Define color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.results)))

        # Plot each model
        for idx, (model_name, data) in enumerate(self.results.items()):
            ax.plot(data['fpr'], data['tpr'],
                    linewidth=2.5,
                    label=f"{model_name} (AUC={data['auc']:.3f})",
                    color=colors[idx])

        ax.set_xlabel('False Positive Rate', fontsize=config.AXIS_LABELSIZE + 2,
                      fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=config.AXIS_LABELSIZE + 2,
                      fontweight='bold')
        ax.set_title('ROC Curves - All Models',
                     fontsize=config.FONT_SIZE + 4, fontweight='bold')
        ax.legend(loc='lower right', fontsize=config.FONT_SIZE,
                  framealpha=0.95, edgecolor='black', ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=config.TICK_LABELSIZE + 2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"All ROC curves saved to {save_path}")

        return fig

    def plot_roc_comparison(self, save_path=None):
        """
        Create a comprehensive ROC comparison figure with both grouped and combined views
        """
        # Create grouped ROC curves (1x3)
        self.plot_roc_curves_grouped(save_path=save_path)

        # Also create combined view
        if save_path:
            combined_path = save_path.replace('.png', '_combined.png')
            self.plot_roc_curves_all(save_path=combined_path)


if __name__ == "__main__":
    # Test ROC visualization
    np.random.seed(42)

    # Create visualizer
    viz = ROCVisualizer()

    # Add simulated results for 9 models
    model_names = ['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'Attention-LSTM',
                   'CNN', 'Transformer', 'MLP', 'Proposed Model']

    for model_name in model_names:
        n_samples = 500
        y_true = np.random.randint(0, 2, n_samples)
        # Simulate different model performances
        noise = np.random.randn(n_samples) * 0.2
        y_pred_prob = np.clip(y_true + noise, 0, 1)
        viz.add_model_results(model_name, y_true, y_pred_prob)

    # Create grouped ROC curves
    groups = [
        ['LSTM', 'GRU', 'BiLSTM'],
        ['CNN-LSTM', 'Attention-LSTM', 'CNN'],
        ['Transformer', 'MLP', 'Proposed Model']
    ]
    viz.plot_roc_curves_grouped(groups=groups, save_path='/tmp/roc_curves_grouped.png')

    # Create combined ROC curves
    viz.plot_roc_curves_all(save_path='/tmp/roc_curves_all.png')

    print("ROC visualizations created successfully!")
