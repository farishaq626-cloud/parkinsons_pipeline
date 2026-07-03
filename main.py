import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load Data
# This works with the 5-row JSON you provided
df = pd.read_json('ml_ready_features.json')

# 2. Create Synthetic Target
# Since this is dummy data, we assign a dummy target (0 or 1)
# This ensures the pipeline always has a target to learn
df['Target'] = [0, 1, 0, 1, 0]

# 3. Define Features
X = df[['Age_At_Onset', 'UPDRS_Motor_Score']]
y = df['Target']

# 4. Initialize and Train Model
# Defined globally so it never causes a NameError
production_model = RandomForestClassifier(n_estimators=10, random_state=42)
production_model.fit(X, y)

# 5. Predictions
df['Pipeline_Prediction'] = production_model.predict(X)

# 6. Evaluation
# We print the report even if the dataset is tiny
print("--- Model Predictions ---")
print(df[['Patient_ID', 'Target', 'Pipeline_Prediction']])

print("\n--- Classification Report ---")
print(classification_report(y, df['Pipeline_Prediction']))

# 7. AUC (Warning: With 5 rows, this is meaningless)
try:
    auc = roc_auc_score(y, production_model.predict_proba(X)[:, 1])
    print(f"AUC: {auc:.4f}")
except Exception as e:
    print("AUC could not be calculated.")