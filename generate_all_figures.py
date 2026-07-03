import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# 1. Load the data
df = pd.read_json('ml_ready_features.json')

# 2. Prepare Data (using the columns we found)
# Features
X = df[['Age_At_Onset']] 
# Target (Binarizing UPDRS_Motor_Score to create a classification target)
y = (df['UPDRS_Motor_Score'] > df['UPDRS_Motor_Score'].median()).astype(int)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Generate ROC Curve
y_pred_proba = model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.pdf')
print("Generated: roc_curve.pdf")

# 5. Generate Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
plt.barh(['Age_At_Onset'], importances, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.pdf')
print("Generated: feature_importance.pdf")