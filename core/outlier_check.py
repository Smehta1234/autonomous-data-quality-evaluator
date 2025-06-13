import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import json

def check_outliers(df: pd.DataFrame, z_thresh: float = 3.0, save_path: str = None):
    report = {
        "zscore": {},
        "isolation_forest": {}
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
        return report

    # === 1. Z-score method ===
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10 or series.std() == 0:
            continue

        z_scores = (series - series.mean()) / series.std()
        outlier_indices = z_scores[abs(z_scores) > z_thresh].index.tolist()

        if outlier_indices:
            report["zscore"][col] = {
                "outlier_count": len(outlier_indices),
                "percent_outliers": round(len(outlier_indices) / len(df) * 100, 2),
                "example_indices": outlier_indices[:10]
            }

    # === 2. Isolation Forest ===
    try:
        clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        preds = clf.fit_predict(df[numeric_cols].fillna(0))
        outlier_rows = np.where(preds == -1)[0].tolist()

        report["isolation_forest"] = {
            "outlier_count": len(outlier_rows),
            "percent_outliers": round(len(outlier_rows) / len(df) * 100, 2),
            "example_indices": outlier_rows[:10]
        }
    except Exception as e:
        report["isolation_forest"] = {"error": str(e)}

    # === Save if requested ===
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)

        if report["zscore"]:
            plot_outlier_distribution(df, report["zscore"])  # <- ADD HERE


import matplotlib.pyplot as plt
import seaborn as sns

def plot_outlier_distribution(df, outlier_cols, save_path="reports/outlier_plot.png"):
    if not outlier_cols:
        return
    plt.figure(figsize=(10, 5))
    scores = [(col, data["percent_outliers"]) for col, data in outlier_cols.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*scores[:10])  # Top 10
    sns.barplot(x=list(values), y=list(labels), palette="rocket")
    plt.xlabel("% Outliers")
    plt.title("Top Features by Outlier % (Z-Score)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


