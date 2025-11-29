"""
Grad-CAM heatmap visualization for understanding what models learn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
import config

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = config.FONT_FAMILY
matplotlib.rcParams['font.size'] = config.FONT_SIZE


class GradCAMVisualizer:
    """Visualize Grad-CAM heatmaps for deep learning models"""

    def __init__(self):
        self.heatmaps = {}

    def compute_gradcam(self, model, input_data, layer_name=None, class_idx=0):
        """
        Compute Grad-CAM heatmap for a given model and input

        Args:
            model: Keras model
            input_data: Input data (single sample)
            layer_name: Name of the convolutional/LSTM layer to visualize
            class_idx: Class index for classification

        Returns:
            Heatmap as numpy array
        """
        # Find the target layer
        if layer_name is None:
            # Try to find the last LSTM, Conv1D, or similar layer
            for layer in reversed(model.layers):
                if isinstance(layer, (keras.layers.LSTM, keras.layers.GRU,
                                      keras.layers.Conv1D, keras.layers.Dense)):
                    layer_name = layer.name
                    break

        if layer_name is None:
            print("Could not find suitable layer for Grad-CAM")
            return None

        # Create gradient model
        try:
            grad_model = keras.models.Model(
                [model.inputs],
                [model.get_layer(layer_name).output, model.output]
            )
        except:
            print(f"Could not create gradient model for layer {layer_name}")
            return None

        # Compute gradients
        with tf.GradientTape() as tape:
            layer_output, predictions = grad_model(input_data)
            loss = predictions[:, class_idx] if len(predictions.shape) > 1 else predictions

        # Get gradients
        grads = tape.gradient(loss, layer_output)

        # Pool gradients
        if grads is not None:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

            # Weight feature maps
            layer_output = layer_output[0]
            for i in range(pooled_grads.shape[-1]):
                layer_output[:, i] *= pooled_grads[i]

            # Create heatmap
            heatmap = tf.reduce_mean(layer_output, axis=-1)
            heatmap = np.maximum(heatmap, 0)  # ReLU
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)  # Normalize

            return heatmap.numpy()
        else:
            return None

    def add_model_heatmap(self, model_name, model, input_samples, layer_names=None):
        """
        Compute and store heatmaps for a model

        Args:
            model_name: Name of the model
            model: Keras model
            input_samples: Input samples to visualize
            layer_names: List of layer names to visualize
        """
        if layer_names is None:
            # Auto-detect layers
            layer_names = []
            for layer in model.layers:
                if isinstance(layer, (keras.layers.LSTM, keras.layers.GRU,
                                      keras.layers.Conv1D)):
                    layer_names.append(layer.name)

        heatmaps_list = []
        for sample_idx in range(min(5, len(input_samples))):  # Max 5 samples
            sample = input_samples[sample_idx:sample_idx+1]
            sample_heatmaps = {}

            for layer_name in layer_names[:5]:  # Max 5 layers
                heatmap = self.compute_gradcam(model, sample, layer_name)
                if heatmap is not None:
                    sample_heatmaps[layer_name] = heatmap

            heatmaps_list.append(sample_heatmaps)

        self.heatmaps[model_name] = heatmaps_list

    def plot_heatmaps_grid(self, n_rows=3, n_cols=5, save_path=None):
        """
        Plot Grad-CAM heatmaps in a grid layout

        Args:
            n_rows: Number of rows
            n_cols: Number of columns
            save_path: Path to save the figure
        """
        if not self.heatmaps:
            print("No heatmaps to display")
            return None

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        axes = axes.flatten()

        plot_idx = 0
        for model_name, heatmaps_list in self.heatmaps.items():
            for sample_idx, sample_heatmaps in enumerate(heatmaps_list):
                if plot_idx >= n_rows * n_cols:
                    break

                if not sample_heatmaps:
                    continue

                # Get the first heatmap
                layer_name, heatmap = list(sample_heatmaps.items())[0]

                ax = axes[plot_idx]
                im = ax.imshow(heatmap.reshape(1, -1), cmap='jet', aspect='auto')
                ax.set_title(f'{model_name}\nSample {sample_idx+1}',
                             fontsize=config.FONT_SIZE, fontweight='bold')
                ax.set_xlabel('Time Steps', fontsize=config.AXIS_LABELSIZE)
                ax.set_ylabel('Activation', fontsize=config.AXIS_LABELSIZE)
                ax.tick_params(labelsize=config.TICK_LABELSIZE)

                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, n_rows * n_cols):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Grad-CAM heatmaps saved to {save_path}")

        return fig

    def plot_layer_activations(self, model_name, sample_idx=0, save_path=None):
        """
        Plot activations for all layers of a specific sample

        Args:
            model_name: Name of the model
            sample_idx: Index of the sample to visualize
            save_path: Path to save the figure
        """
        if model_name not in self.heatmaps:
            print(f"No heatmaps found for {model_name}")
            return None

        sample_heatmaps = self.heatmaps[model_name][sample_idx]
        n_layers = len(sample_heatmaps)

        if n_layers == 0:
            print(f"No layer heatmaps available for {model_name}")
            return None

        fig, axes = plt.subplots(n_layers, 1, figsize=(12, n_layers*2))
        if n_layers == 1:
            axes = [axes]

        for idx, (layer_name, heatmap) in enumerate(sample_heatmaps.items()):
            ax = axes[idx]
            im = ax.imshow(heatmap.reshape(1, -1), cmap='jet', aspect='auto')
            ax.set_title(f'{layer_name}', fontsize=config.FONT_SIZE, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=config.AXIS_LABELSIZE)
            ax.tick_params(labelsize=config.TICK_LABELSIZE)
            plt.colorbar(im, ax=ax)

        plt.suptitle(f'{model_name} - Sample {sample_idx+1} - Layer Activations',
                     fontsize=config.FONT_SIZE + 2, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"Layer activations saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Test Grad-CAM visualization
    from models.lstm_model import build_lstm_model

    print("Building test model...")
    model = build_lstm_model((config.SEQUENCE_LENGTH, len(config.FEATURES)))

    # Create simulated input
    input_samples = np.random.randn(10, config.SEQUENCE_LENGTH, len(config.FEATURES))

    # Create visualizer
    viz = GradCAMVisualizer()

    print("Computing Grad-CAM heatmaps...")
    viz.add_model_heatmap("LSTM_Test", model, input_samples)

    print("Plotting heatmaps...")
    viz.plot_heatmaps_grid(save_path='/tmp/gradcam_grid.png')

    print("Grad-CAM visualization created successfully!")
