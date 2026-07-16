import pandas as pd
from sklearn.metrics import roc_auc_score
# Assuming production_model and X are already in memory or loaded

# 1. AUC Score
auc_score = roc_auc_score(y, production_model.predict_proba(X)[:, 1])
print(f"AUC Value: {auc_score:.3f}")

# 2. Feature Importance
importances = production_model.feature_importances_
feature_names = ['Age_At_Onset', 'Progression_Velocity']
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nTop Features:")
print(importance_df.sort_values(by='Importance', ascending=False))