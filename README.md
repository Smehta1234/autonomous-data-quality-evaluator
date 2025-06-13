# 🧠 Autonomous Data Quality Evaluator (ADQE)

**ADQE** is a smart tool that automatically analyzes your dataset for common data issues — before you even start modeling. It scans your CSV for leakage, missing values, drift, outliers, bias, and more — then gives you clear summaries and plots in a sleek Streamlit dashboard.

---

## 🚀 What It Detects

| Check                | Description                                                 |
|---------------------|-------------------------------------------------------------|
| Missing Values       | Shows % missing per column + plot                          |
| Data Leakage         | Detects features that directly leak the target             |
| Data Drift           | Compares training vs test distribution                     |
| Outliers             | Z-score + Isolation Forest outliers + top example indices  |
| Redundancy           | Correlated or low-variance features                        |
| Bias / Imbalance     | Warns on skewed class distributions                        |

---

## 🖼️ Live Demo 

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autonomous-data-quality-evaluator-rwtqdgavmcgtcgsazauyzp.streamlit.app/)

---

## 💻 Local Installation

Clone the repo and run Streamlit:

```bash
git clone https://github.com/Smehta1234/autonomous-data-quality-evaluator.git
cd autonomous-data-quality-evaluator
pip install -r requirements.txt
streamlit run app.py
