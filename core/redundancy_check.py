import pandas as pd
import numpy as np
import os
import json

def check_redundancy(df: pd.DataFrame, corr_thresh: float = 0.95, save_path: str = None):
    report = {
        "high_correlation": {},
        "low_variance": []
    }

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # === 1. Highly Correlated Features ===
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper_triangle.columns:
        correlated = upper_triangle[col][upper_triangle[col] > corr_thresh]
        for other_col, corr_val in correlated.items():
            report["high_correlation"][f"{other_col} ↔ {col}"] = round(corr_val, 4)

    # === 2. Constant / Low-Variance Features ===
    for col in df.columns:
        if df[col].nunique() <= 1:
            report["low_variance"].append(col)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)

        plot_correlation_heatmap(numeric_df)

    return report

def plot_correlation_heatmap(df, save_path="reports/redundancy_corr.png"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


