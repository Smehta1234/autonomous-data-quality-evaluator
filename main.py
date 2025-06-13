from core.main_analyzer import analyze_dataset

# Full dataset path
train_path = "data/UCI_Credit_Card.csv"

# Simulate a test set by splitting randomly
import pandas as pd
df = pd.read_csv(train_path)
df.sample(frac=0.5, random_state=42).to_csv("data/test.csv", index=False)

# Analyze full pipeline
analyze_dataset(
    path=train_path,
    target_col="default.payment.next.month",       # the label column in this dataset
    test_path="data/test.csv"
)



