"""
Model evaluation and metrics calculation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


class ModelEvaluator:
    """Evaluate model performance and calculate comprehensive metrics"""

    def __init__(self):
        self.results = {}

    @staticmethod
    def calculate_mape(y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAPE value (in percentage)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def calculate_metrics(self, y_true, y_pred, y_pred_prob):
        """
        Calculate comprehensive metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_prob: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_true, y_pred_prob),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['Confusion_Matrix'] = cm

        # Additional metrics
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['TN'] = tn
            metrics['FP'] = fp
            metrics['FN'] = fn
            metrics['TP'] = tp
            metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def add_model_results(self, model_name, y_true, y_pred, y_pred_prob):
        """
        Add evaluation results for a model

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_prob: Predicted probabilities
        """
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_prob)
        self.results[model_name] = metrics
        return metrics

    def create_comparison_table(self):
        """
        Create a comparison table similar to the paper's Table II

        Returns:
            pandas DataFrame with model comparison
        """
        if not self.results:
            print("No results to display")
            return None

        # Create DataFrame
        df_data = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'MAE': f"{metrics['MAE']:.4f}",
                'MAPE(%)': f"{metrics['MAPE']:.2f}",
                'RMSE': f"{metrics['RMSE']:.4f}",
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1-Score': f"{metrics['F1-Score']:.4f}",
                'AUC': f"{metrics['AUC']:.4f}",
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Sort by AUC (descending)
        df = df.sort_values('AUC', ascending=False, ignore_index=True)

        return df

    def save_comparison_table(self, filepath):
        """Save comparison table to CSV"""
        df = self.create_comparison_table()
        if df is not None:
            df.to_csv(filepath, index=False)
            print(f"Comparison table saved to {filepath}")
        return df

    def print_comparison_table(self):
        """Print formatted comparison table"""
        df = self.create_comparison_table()
        if df is not None:
            print("\n" + "="*120)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*120)
            print(df.to_string(index=False))
            print("="*120)
        return df

    def get_best_model(self, metric='AUC'):
        """
        Get the best performing model based on a metric

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.results:
            return None, None

        best_model = None
        best_value = -float('inf') if metric != 'MAE' else float('inf')

        for model_name, metrics in self.results.items():
            value = metrics.get(metric, None)
            if value is None:
                continue

            if metric in ['MAE', 'RMSE', 'MAPE']:
                # Lower is better
                if value < best_value:
                    best_value = value
                    best_model = model_name
            else:
                # Higher is better
                if value > best_value:
                    best_value = value
                    best_model = model_name

        return best_model, best_value


if __name__ == "__main__":
    # Test evaluator
    np.random.seed(42)

    # Simulated data
    y_true = np.random.randint(0, 2, 100)
    y_pred_prob = np.random.rand(100)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Create evaluator
    evaluator = ModelEvaluator()

    # Add results for multiple models
    evaluator.add_model_results("Model A", y_true, y_pred, y_pred_prob)
    evaluator.add_model_results("Model B", y_true, y_pred * 0.9, y_pred_prob * 0.9)
    evaluator.add_model_results("Model C", y_true, y_pred * 0.8, y_pred_prob * 0.8)

    # Print comparison table
    evaluator.print_comparison_table()

    # Get best model
    best_model, best_auc = evaluator.get_best_model('AUC')
    print(f"\nBest model: {best_model} (AUC = {best_auc:.4f})")
