import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
)

logging.basicConfig(level = logging.INFO, format = "%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

def display_classification_metrics(y_true, y_pred, y_proba,*, target_names = None, threshold = None):

    if threshold is not None:
        logger.info(f"\nClassification metrics at threshold: {threshold:.2f}")

    logger.info("\n--- Classification Metrics ---")
    logger.info("Classification Report:")
    if target_names is not None:
        report = classification_report(y_true, y_pred, target_names = target_names)
    else:
        report = classification_report(y_true, y_pred)
    logger.info(report)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'binary')
    recall = recall_score(y_true, y_pred, average = 'binary')
    f1 = f1_score(y_true, y_pred, average = 'binary')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")



def plot_confusion(y_true, y_pred, title = None, target_names = None, figsize=(6, 5), model_name = None, filepath = None):
    """
    Plots a confusion matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix" or customized with model_name.
        target_names (list, optional): List of target class names (e.g., ['No Recession', 'Recession']).
                                        If None, uses unique sorted labels from y_true.
        figsize (tuple, optional): Figure size. Defaults to (6, 5).
        model_name (str, optional): Model name to include in the plot title.
        filepath (str, optional): If given, save the plot to this path instead of showing it.
    """

    cm = confusion_matrix(y_true, y_pred)
    if target_names is None:
        target_names = [str(x) for x in sorted(np.unique(y_true))]

    plt.figure(figsize = figsize)
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', cbar = False, xticklabels = target_names, yticklabels = target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if title is None:
        if model_name:
            plt.title(f"Confusion Matrix - {model_name}")
        else:
            plt.title("Confusion Matrix")
    else:
        plt.title(title)

    if filepath:
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {filepath}")
    else:
        plt.show()
        logger.info(f"Displayed confusion matrix plot{f' for {model_name}' if model_name else ''}.")


def plot_roc_pr_curves(y_true, y_proba, title = None, figsize=(12, 5), model_name = None, filepath = None):
    """
    Plots ROC and Precision-Recall curves.

    Args:
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        title (str, optional): Overall title for the plots. Defaults to "ROC and Precision-Recall Curves" or customized with model_name.
        figsize (tuple, optional): Figure size. Defaults to (12, 5).
        model_name (str, optional): Model name to include in the plot title.
        filepath (str, optional): If given, save the plot to this path instead of showing it.
    """

    if title is None:
        title = f"ROC and Precision-Recall Curves - {model_name}" if model_name else "ROC and Precision-Recall Curves"

    fig, axes = plt.subplots(1, 2, figsize = figsize)
    fig.suptitle(title, fontsize = 14)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color = 'darkorange', lw = 2, label = f'ROC curve (area = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].legend(loc = "lower right")

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    axes[1].plot(recall, precision, color = 'blue', lw = 2, label = f'PR curve (area = {avg_precision:.2f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc = "lower left")

    plt.tight_layout(rect = [0, 0.03, 1, 0.95])

    if filepath:
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Saved ROC and PR curves plot to {filepath}")
    else:
        plt.show()
        logger.info(f"Displayed ROC and PR curves plot{f' for {model_name}' if model_name else ''}.")


def summarize_feature_importance(model_pipeline, importances_list, feature_names, top_n = 10):
    """
    Summarizes and displays feature importances (or coefficients) from a list of models
    within a pipeline. Handles cases where feature names might change after preprocessing.

    Args:
        model_pipeline (sklearn.pipeline.Pipeline): The trained scikit-learn pipeline.
        importances_list (list of array-like): A list where each element is an array/list
                                              of feature importances or coefficients for one model.
                                              For Logistic Regression, it would be `[model.coef_[0]]`.
                                              For tree models, `[model.feature_importances_]`.
        feature_names (list): The list of original feature names before any pipeline preprocessing.
                              Typically, X_train.columns.tolist().
        top_n (int, optional): Number of top features to display. Defaults to 20.
    """
    if not importances_list:
        logger.warning("No feature importances or coefficients provided for summarization.")
        return

    all_model_importances = []

    preprocessor_step = None
    if 'preprocessor' in model_pipeline.named_steps:
        preprocessor_step = model_pipeline.named_steps['preprocessor']

    for i, importances in enumerate(importances_list):
        model_name = f"Model_{i+1}" 
        processed_feature_names = feature_names 
        if preprocessor_step and hasattr(preprocessor_step, 'get_feature_names_out'):
            try:
                processed_feature_names = preprocessor_step.get_feature_names_out(feature_names)
            except Exception as e:
                logger.warning(f"Could not get processed feature names from preprocessor: {e}. Falling back to original names.")
                processed_feature_names = feature_names
        elif preprocessor_step and hasattr(preprocessor_step, 'steps'): 
             pass
        
        if len(importances) != len(processed_feature_names):
            logger.warning(f"Length mismatch for {model_name}: {len(importances)} importances but {len(processed_feature_names)} processed features. Skipping this model for aggregation.")
            continue

        importance_series = pd.Series(importances, index = processed_feature_names, name = model_name)
        all_model_importances.append(importance_series)

    if not all_model_importances:
        logger.warning("No valid feature importances were collected after processing. Cannot summarize.")
        return

    importances_df = pd.concat(all_model_importances, axis=1)
    importances_df_filled = importances_df.fillna(0)
    normalized_importances = importances_df_filled.apply(lambda col: col / (col.sum() + 1e-9), axis = 0)
    mean_importance = normalized_importances.mean(axis = 1).sort_values(ascending = False)

    print("\n--- Aggregated Feature Importance (Mean Normalized Score) ---")
    print(mean_importance.head(top_n))

    plt.figure(figsize = (10, min(max(6, top_n * 0.4), 15)))
    sns.barplot(x = mean_importance.head(top_n).values, y = mean_importance.head(top_n).index, palette = 'viridis')
    plt.title(f'Top {top_n} Aggregated Feature Importances')
    plt.xlabel('Mean Normalized Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    logger.info(f"Generated Top {top_n} Aggregated Feature Importances plot.")