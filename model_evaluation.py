# model evaluation and threshold optimization

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_models(
    trained_pipelines,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    display_classification_metrics,
    plot_confusion,
    plot_roc_pr_curves,
    output_path = "models/final_model_results_summary.csv"
):
    """
    Evaluate trained models, tune thresholds, compute metrics, and save results.

    Parameters:
        trained_pipelines (dict): Dict of model_name -> trained pipeline
        best_params (dict): Dict of model_name -> best hyperparameters
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        display_classification_metrics (function): Function to display classification metrics
        plot_confusion (function): Function to plot confusion matrix
        plot_roc_pr_curves (function): Function to plot ROC & PR curves
        output_path (str): Path to save final results summary CSV
    """
    all_model_results_summary = {}

    for model_name, best_pipeline in trained_pipelines.items():
        print(f"\n--- Evaluating {model_name} ---")

        y_train_proba = best_pipeline.predict_proba(X_train)[:, 1]
        y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

        # Threshold tuning
        logger.info(f"Tuning threshold for {model_name} on X_train (F1-score optimization).")
        thresholds = np.linspace(0.01, 0.99, 100)
        f1_scores_train = [
            f1_score(y_train, (y_train_proba >= t).astype(int), zero_division = 0)
            for t in thresholds
        ]

        best_threshold = 0.5
        best_f1_train = 0.0

        if np.max(f1_scores_train) > 0:
            best_threshold = thresholds[np.argmax(f1_scores_train)]
            best_f1_train = np.max(f1_scores_train)
            logger.info(
                f"Optimal threshold for {model_name}: {best_threshold:.2f} "
                f"(F1 on train: {best_f1_train:.4f})"
            )
        else:
            logger.warning(f"No optimal threshold found for {model_name}. Defaulting to 0.5.")
            logger.warning(f"Check y_train and y_train_proba for {model_name} for issues.")

        print(f"Best Threshold for {model_name}: {best_threshold:.2f} | F1 Train: {best_f1_train:.4f}")

        # Final predictions
        y_train_pred = (y_train_proba >= best_threshold).astype(int)
        y_test_pred = (y_test_proba >= best_threshold).astype(int)

        print(f"\n--- Performance of {model_name} on Test Set ({X_test.index.min().date()} to {X_test.index.max().date()}) ---")

        # Metrics
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division = 0),
            "recall": recall_score(y_test, y_test_pred, zero_division = 0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division = 0),
            "roc_auc_score": roc_auc_score(y_test, y_test_proba),
            "average_precision_score": average_precision_score(y_test, y_test_proba),
        }

        display_classification_metrics(y_true = y_test,y_pred = y_test_pred,y_proba = y_test_proba,threshold = best_threshold)
        plot_confusion(y_test, y_test_pred, model_name = model_name, filepath = f"plots/{model_name}_confusion_matrix.png")
        plot_roc_pr_curves(y_test, y_test_proba, model_name = model_name, filepath = f"plots/{model_name}_roc_pr_curves.png")

        all_model_results_summary[model_name] = {
            "best_threshold_tuned_on_train": best_threshold,
            "f1_score_on_test_with_tuned_threshold": metrics["f1_score"],
            "metrics": metrics,
            "best_params": best_params.get(model_name, {}),
        }

    results_for_df = []
    for model, results in all_model_results_summary.items():
        row = {
            "model": model,
            "best_threshold_tuned_on_train": results["best_threshold_tuned_on_train"],
            "f1_score_on_test_with_tuned_threshold": results["f1_score_on_test_with_tuned_threshold"],
            **results["metrics"],
        }
        results_for_df.append(row)

    results_df = pd.DataFrame(results_for_df)
    results_df.to_csv(output_path, index = False)
    print(f"\nSaved all model summaries to '{output_path}'")

    logger.info("Evaluation complete.")
    return results_df

if __name__ == "__main__":
    raise RuntimeError("This module is intended to be imported, not executed directly.")
