# data_viz.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

os.makedirs("plots", exist_ok=True)

logging.basicConfig(level = logging.INFO,format = "%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

def save_and_show_plot(filename):
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi = 300)
    plt.show()

def plot_recession_class_distribution(df):
    sns.set_style("whitegrid")
    colors = ['green', 'red']
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data = df, x='recession', hue = 'recession', palette = colors, legend = False)
    plt.xlabel("Recession Class")
    plt.ylabel("Count")
    plt.title("Recession Class Distribution")
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2, count), 
                    ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')
    save_and_show_plot("recession_class_distribution")

def plot_correlation_heatmap(df, threshold = 0.15):
    sns.set(style = "white")
    corr_target = df.corr(numeric_only = True)['recession'].sort_values(ascending = True)
    logging.info("\nCorrelation of features with recession:\n%s", corr_target)
    top_corr_features = corr_target[abs(corr_target) > threshold].index.tolist()
    if 'recession' not in top_corr_features:
        top_corr_features.append('recession')
    plt.figure(figsize = (12, 8))
    sns.heatmap(df[top_corr_features].corr(), annot = True, cmap = sns.diverging_palette(10, 240, as_cmap = True))
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    plt.title("Correlation Matrix of Top Features")
    save_and_show_plot("top_features_correlation_heatmap")

def plot_recession_timeline(df):
    sns.set_style("whitegrid")
    plt.figure(figsize = (14, 4))
    sns.lineplot(data = df, x = 'date', y = 'recession', drawstyle = 'steps-post', color = 'red')
    plt.fill_between(df['date'], df['recession'], step = 'post', alpha = 0.5, color = 'blue')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['No Recession', 'Recession'])
    plt.xlabel("Date")
    plt.ylabel("")
    plt.title("U.S. Recession Timeline")
    plt.grid(axis = 'y', linestyle = '--', alpha = 1)
    save_and_show_plot("us_recession_timeline")

def plot_feature_over_time(df, feature, highlight_recession = True):
    sns.set_style("whitegrid")
    plt.figure(figsize = (14, 5))
    ax = sns.lineplot(data = df, x = 'date', y = feature, label = feature)
    if highlight_recession and 'recession' in df.columns:
        recession_periods = df[df['recession'] == 1]
        for _, row in recession_periods.iterrows():
            plt.axvspan(row['date'], row['date'], color = 'red', alpha = 0.05)
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.title(f"{feature} Over Time")
    plt.grid(True)
    save_and_show_plot(f"{feature}_trend")

if __name__ == "__main__":
    raise RuntimeError("This module is intended to be imported, not executed directly.")
