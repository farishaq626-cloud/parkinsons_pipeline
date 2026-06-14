import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("🧠 Parkinson's Pipeline: Initializing Predictive Machine Learning Layer...\n")

# 1. Simulating a larger, scaled clinical cohort 
# In a real scenario, this pulls directly from your 'ml_ready_features.json'
np.random.seed(42) # Keeps splits consistent
n_patients = 100

# Generating mock scaled features (Mean=0, Var=1) and a binary progression target
# Target: 1 = Rapid Progression (High Risk), 0 = Stable (Low Risk)
mock_data = {
    'Age_Scaled': np.random.normal(0, 1, n_patients),
    'UPDRS_Motor_Scaled': np.random.normal(0, 1, n_patients),
    'Rapid_Progression': np.random.choice([0, 1], size=n_patients, p=[0.6, 0.4])
}

df_ml = pd.DataFrame(mock_data)
X = df_ml[['Age_Scaled', 'UPDRS_Motor_Scaled']]
y = df_ml['Rapid_Progression']

print(f"📥 Dataset prepared for training. Total cohort size: {len(df_ml)} patients.")
print(f"📊 Class distribution: {np.bincount(y)[0]} Stable vs. {np.bincount(y)[1]} Rapid Progressors.\n")


# 2. Train/Test Split (Ensuring validation integrity)
# 80% of data trains the model, 20% is held back to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"✂️ Data split complete: {len(X_train)} training samples, {len(X_test)} validation samples.")


# 3. Model Training
print("🏋️‍♂️ Training Logistic Regression Classifier...")
model = LogisticRegression(class_weight='balanced' ,random_state=42)
model.fit(X_train, y_train)


# 4. Evaluation and Performance Metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 MODEL EVALUATION METRICS (Validation Set):")
print(f"Overall Accuracy: {accuracy * 100:.1f}%")
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stable (0)', 'Rapid (1)']))

# Save out model weights summary or predictions
df_test_results = X_test.copy()
df_test_results['Actual_Progression'] = y_test
df_test_results['Predicted_Progression'] = y_pred
df_test_results.to_json('model_predictions_output.json', orient='records', indent=4)
print("💾 Success: Evaluation complete. Predictions saved to 'model_predictions_output.json'.")