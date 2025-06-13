import pandas as pd
import os
import json

from core.missing_check import check_missing_values
from core.leakage_check import check_leakage
from core.drift_check import check_drift
from core.outlier_check import check_outliers
from core.redundancy_check import check_redundancy
from core.bias_check import check_bias

def analyze_dataset(path: str, target_col: str = None, test_path: str = None):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    os.makedirs("reports", exist_ok=True)

    # === 1. Missing Values ===
    print("\n[1] Checking Missing Values...")
    missing_report = check_missing_values(df, save_path="reports/missing_report.json")
    print(f"Saved missing value report with {len(missing_report)} flagged columns.")

    # === 2. Leakage Detection ===
    if target_col:
        print("\n[2] Checking for Leakage...")
        leakage_report = check_leakage(df, target_col=target_col)
        with open("reports/leakage_report.json", 'w') as f:
            json.dump(leakage_report, f, indent=4)
        print(f"Saved leakage report with {len(leakage_report)} suspect columns.")

    # === 3. Drift Detection ===
    if test_path:
        print("\n[3] Checking Drift with test set:", test_path)
        test_df = pd.read_csv(test_path)
        drift_report = check_drift(df, test_df, save_path="reports/drift_report.json")
        print(f"Saved drift report with {len(drift_report)} compared columns.")

    # === 4. Outlier Detection ===
    print("\n[4] Checking Outliers...")
    outlier_report = check_outliers(df, save_path="reports/outlier_report.json")
    print(f"Saved outlier report.")

    # === 5. Redundancy Detection ===
    print("\n[5] Checking Redundancy...")
    redundancy_report = check_redundancy(df, save_path="reports/redundancy_report.json")
    print(f"Saved redundancy report with {len(redundancy_report['high_correlation'])} correlated pairs.")

    # === 6. Bias Detection ===
    print("\n[6] Checking Bias...")
    bias_report = check_bias(df, target_col=target_col, save_path="reports/bias_report.json")
    print(f"Saved bias report with {len(bias_report)} flagged columns.")

    print("\n Analysis complete. Check the /reports folder.")
