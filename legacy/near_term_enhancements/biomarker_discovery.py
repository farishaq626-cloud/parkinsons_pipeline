from sklearn.ensemble import RandomForestClassifier
import pandas as pd
def get_top_biomarkers(df, target):
    X = df.drop(columns=["patient_id", target])
    y = df[target]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    importances = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    return importances.sort_values(by="importance", ascending=False).head(10), model