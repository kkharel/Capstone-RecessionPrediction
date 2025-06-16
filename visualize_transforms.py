import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from statsmodels.graphics.tsaplots import plot_acf

logging.basicConfig(level = logging.INFO,format = "%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

def create_output_dirs(base_dir = "plots", transformations = None):
    """Creates folders to save plots for each transformation."""
    os.makedirs(base_dir, exist_ok = True)
    if transformations is None:
        transformations = ["original", "log", "diff", "log_diff"]
    for t in transformations:
        os.makedirs(os.path.join(base_dir, t), exist_ok = True)

def save_and_plot(series, title, ax, kind = 'line', lags = 15):
    """Helper to plot and title the figure."""
    if kind == 'line':
        ax.plot(series)
        ax.set_title(title)
        ax.grid(True)
    elif kind == 'hist':
        ax.hist(series.dropna(), bins = 30)
        ax.set_title(f"Histogram: {title}")
    elif kind == 'acf':
        plot_acf(series.dropna(), lags = lags, ax = ax)
        ax.set_title(f"ACF: {title}")

def plot_transformations_for_column(series, column_name, base_dir = "plots", lags = 10, run_transforms = None):
    """Generates and saves plots for selected transformations for a column."""
    transforms = {
        "original": series,
        "log": np.log(series + 1e-6) if (series > 0).all() else None,
        "diff": series.diff(),
        "log_diff": np.log(series + 1e-6).diff() if (series > 0).all() else None
    }

    for key, ts in transforms.items():
        if run_transforms and key not in run_transforms:
            logging.info(f"Skipping {key} transformation for {column_name} as it is not in the run_transforms list.")
            continue
        if ts is None or ts.dropna().empty:
            logging.warning(f"Skipping {key} transformation for {column_name}: series is empty or contains invalid values.")
            continue

        fig, axs = plt.subplots(1, 3, figsize = (16, 4))
        fig.suptitle(f"{column_name} — {key} Transformation", fontsize = 14)

        save_and_plot(ts, f"{key} - Line", axs[0], kind = 'line')
        save_and_plot(ts, f"{key} - Histogram", axs[1], kind = 'hist')
        save_and_plot(ts, f"{key} - ACF", axs[2], kind = 'acf', lags = lags)

        plt.tight_layout(rect = [0, 0, 1, 0.95])
        save_path = os.path.join(base_dir, key, f"{column_name}_{key}.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved {key} plots for {column_name} to {save_path}")

def visualize_transformations_on_df(df, exclude_cols = None, base_dir = "plots", run_transforms = None, lags = 15):
    """Applies selected transformations and saves plots for all numeric columns."""
    if exclude_cols is None:
        exclude_cols = ['date', 'recession']
    numeric_cols = df.select_dtypes(include = [np.number]).columns
    create_output_dirs(base_dir, transformations = run_transforms)

    for col in numeric_cols:
        if col in exclude_cols:
            logging.info(f"Skipping excluded column: {col}")
            continue
        logging.info(f"Processing column: {col}")
        plot_transformations_for_column(df[col], col, base_dir = base_dir, lags = lags, run_transforms = run_transforms)


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_economic_data.csv", parse_dates = ["date"])
    # Can modify the list to ['original', 'log'] or any subset
    visualize_transformations_on_df(
        df,
        exclude_cols = ["date", "recession"],
        base_dir = "plots/econ_transforms",
        run_transforms = ["original", "log", "diff", "log_diff"],
        lags=15
    )
