import streamlit as st
import pandas as pd
import os
import shutil

from core.main_analyzer import analyze_dataset

# === UI Settings ===
st.set_page_config(page_title="Autonomous Data Quality Evaluator", layout="wide")
st.title("🧠 Autonomous Data Quality Evaluator (ADQE)")

st.markdown("""
Upload a dataset and get automatic insights into common data quality issues that may silently affect your ML model's performance.
""")

# === File Upload ===
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

# === Analysis Form ===
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully.")

    with st.expander("Preview Dataset"):
        st.dataframe(df.head())

    # Auto-detect target column
    default_target = df.columns[-1] if df.columns[-1] not in ["ID", "index"] else None
    target_col = st.selectbox("Select Target Column (optional)", [""] + list(df.columns), index=(df.columns.get_loc(default_target) if default_target else 0))

    simulate_test = st.checkbox("Simulate test split automatically?", value=True)
    run_button = st.button("🚀 Run Analysis")

    if run_button:
        # Save uploaded file temporarily
        os.makedirs("data", exist_ok=True)
        data_path = "data/uploaded.csv"
        df.to_csv(data_path, index=False)

        # Optional test split
        test_path = None
        if simulate_test:
            test_df = df.sample(frac=0.3, random_state=42)
            test_path = "data/test.csv"
            test_df.to_csv(test_path, index=False)

        # Clear old reports
        if os.path.exists("reports"):
            shutil.rmtree("reports")

        # Run analyzer
        with st.spinner("Analyzing your dataset..."):
            analyze_dataset(
                path=data_path,
                target_col=target_col if target_col != "" else None,
                test_path=test_path
            )
        st.success("Analysis complete! Scroll down to see results.")

        # === Results Display ===
        st.header("📊 Summary Reports")


        def show_json_report(title, filename):
            path = f"reports/{filename}"
            if os.path.exists(path):
                with st.expander(f"📂 {title}"):
                    with open(path, 'r') as f:
                        import json
                        report = json.load(f)
                        if not report:
                            st.success("✅ No issues detected.")
                        else:
                            st.json(report)
                    st.download_button(label=f"Download {filename}", data=open(path, "rb"), file_name=filename)


        show_json_report("Missing Value Report", "missing_report.json")
        show_json_report("Leakage Detection Report", "leakage_report.json")
        show_json_report("Drift Detection Report", "drift_report.json")
        show_json_report("Outlier Detection Report", "outlier_report.json")
        show_json_report("Redundancy Report", "redundancy_report.json")
        show_json_report("Bias Report", "bias_report.json")

        # === Plots Display ===
        st.markdown("---")
        st.header("🖼️ Visualizations")

        def show_plot(title, path):
            if os.path.exists(path):
                with st.expander(f"📊 {title}"):
                    st.image(path, use_column_width=True)

        show_plot("Missing Values", "reports/missing_plot.png")
        show_plot("Outlier Distribution", "reports/outlier_plot.png")
        show_plot("Feature Correlation Heatmap", "reports/redundancy_corr.png")

        # Drift plots (loop over files)
        import glob

        for path in sorted(glob.glob("reports/drift_*.png")):
            feature = os.path.basename(path).replace("drift_", "").replace(".png", "")
            show_plot(f"Drift: {feature}", path)

        # Bias plots (loop over files)
        for path in sorted(glob.glob("reports/bias_*.png")):
            feature = os.path.basename(path).replace("bias_", "").replace(".png", "")
            show_plot(f"Bias: {feature}", path)


