import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

def check_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.9):
    report = {}
    y = df[target_col]
    features = df.drop(columns=[target_col])

    for col in features.columns:
        X_col = features[[col]].copy()
        if X_col[col].dtype == 'object':
            X_col = pd.get_dummies(X_col, drop_first=True)

        # drop NA rows for this column only
        valid = X_col.notnull().all(axis=1) & y.notnull()
        if valid.sum() < 50:  # not enough data
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_col[valid], y[valid], test_size=0.3, random_state=42
            )
            model = GradientBoostingClassifier(n_estimators=50, max_depth=3)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)

            if auc >= threshold:
                report[col] = round(auc, 4)
        except Exception as e:
            continue  # skip columns with errors

    return report

