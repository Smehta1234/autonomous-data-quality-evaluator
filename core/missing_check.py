import pandas as pd
import json
import os


import matplotlib.pyplot as plt

def plot_missing(report: dict):
    if not report:
        print("No missing values to plot.")
        return

    cols = list(report.keys())
    ratios = [report[col]["missing_ratio"] for col in cols]

    plt.figure(figsize=(8, 4))
    plt.bar(cols, ratios, color='orange')
    plt.title("Missing Value Ratio per Column")
    plt.ylabel("Missing Ratio")
    plt.xlabel("Feature")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/missing_plot.png")
    plt.close()


def check_missing_values(df: pd.DataFrame, threshold: float = 0.3, save_path: str = None):
    report = {}
    total_rows = len(df)

    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_ratio = round(missing_count / total_rows, 4)
        critical = missing_ratio > threshold

        if missing_count > 0:
            report[col] = {
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "critical": bool(critical)
            }

    # Save report as JSON if path is given
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)

    if save_path:
        plot_missing(report)

    return report



