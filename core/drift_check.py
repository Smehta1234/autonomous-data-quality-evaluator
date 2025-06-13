import pandas as pd
from scipy.stats import ks_2samp
import os
import json

def check_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.1, save_path: str = None):
    report = {}

    shared_cols = set(train_df.columns).intersection(set(test_df.columns))

    for col in shared_cols:
        try:
            train_col = train_df[col].dropna()
            test_col = test_df[col].dropna()

            if train_col.dtype == 'object' or test_col.dtype == 'bool':
                continue

            stat, p_value = ks_2samp(train_col, test_col)
            drifted = p_value < threshold

            report[col] = {
                "ks_stat": float(round(stat, 4)),
                "p_value": float(round(p_value, 4)),
                "drift_detected": bool(drifted)
            }
        except Exception as e:
            continue

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)

        plot_drift(train_df, test_df, report)

    return report

import matplotlib.pyplot as plt
import seaborn as sns

def plot_drift(train, test, report, save_dir="reports/"):
    for feature in report:
        if report[feature]["drift_detected"]:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(train[feature].dropna(), label="Train", fill=True)
            sns.kdeplot(test[feature].dropna(), label="Test", fill=True)
            plt.title(f"Drift Detected: {feature}")
            plt.xlabel(feature)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/drift_{feature}.png")
            plt.close()


