import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# 1. Load the data that your pipeline already successfully saved
# Using the file your terminal confirmed was created
df = pd.read_json('ml_ready_features.json')

# Assuming your json has 'Age_At_Onset', 'Progression_Velocity', and 'Progression_Target'
X = df[['Age_At_Onset', 'Progression_Velocity']]
y = df['Progression_Target']

# 2. Train the model inside this script (independent of main.py)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 3. Figure 2: ROC Curve
y_pred_proba = model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig('roc_curve.pdf')
print("Saved roc_curve.pdf")

# 4. Figure 3: Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
plt.barh(['Age', 'Velocity'], importances, color='skyblue')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.pdf')
print("Saved feature_importance.pdf")