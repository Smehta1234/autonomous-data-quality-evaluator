import pandas as pd
import os
import json

def check_bias(df: pd.DataFrame, target_col: str = None, min_group_pct: float = 0.05, save_path: str = None):
    report = {}

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    for col in categorical_cols:
        group_counts = df[col].value_counts(normalize=True)
        small_groups = group_counts[group_counts < min_group_pct]

        if len(small_groups) > 0:
            report[col] = {
                "underrepresented_groups": small_groups.to_dict(),
                "note": "These groups may be underrepresented and could bias model performance."
            }

        if target_col and col != target_col:
            try:
                cross_tab = pd.crosstab(df[col], df[target_col], normalize='index')
                report[col]["target_distribution"] = cross_tab.to_dict()
            except:
                continue

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)

        plot_bias_distributions(df, list(report.keys()))

    return report

def plot_bias_distributions(df, flagged_cols, save_dir="reports/"):
    import matplotlib.pyplot as plt
    for col in flagged_cols:
        plt.figure(figsize=(6, 4))
        df[col].value_counts(normalize=True).plot(kind="bar", color="orange")
        plt.title(f"Group Distribution for {col}")
        plt.ylabel("Percentage")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bias_{col}.png")
        plt.close()

