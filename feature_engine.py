import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

print("🤖 Parkinson's Pipeline: Initializing Feature Scaling & Preprocessing Layer...\n")

# 1. Simulating a Clean Cohort Dataset
# Age and UPDRS motor scores have completely different scales and ranges.
cohort_data = {
    'Patient_ID': ['P-001', 'P-002', 'P-003', 'P-004', 'P-005'],
    'Age_At_Onset': [55.0, 68.0, 42.0, 71.0, 60.0],
    'UPDRS_Motor_Score': [14.0, 38.5, 9.0, 44.0, 22.5]
}

df_cohort = pd.DataFrame(cohort_data)
print("⚠️ Raw Feature Profiles (Before Scaling):")
print(df_cohort.drop(columns=['Patient_ID']))
print("-"*50 + "\n")


# 2. Mathematical Standard Scaling Layer (Z-score Normalization)
# This centers the features around a mean of 0 and scales them to a standard deviation of 1.
def scale_clinical_features(df):
    features_to_scale = ['Age_At_Onset', 'UPDRS_Motor_Score']
    
    # Initialize the Scikit-Learn StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to our metrics and transform them into a normalized matrix
    scaled_matrix = scaler.fit_transform(df[features_to_scale])
    
    # Map the scaled matrix back into a clean DataFrame wrapper
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaled_matrix
    
    print("📈 Feature Scaling Complete: Z-score Normalization applied successfully.")
    return df_scaled

# Run the preprocessing engine
df_processed_features = scale_clinical_features(df_cohort)


# 3. Display the Normalised Array Output
print("\n🎯 Standardised Machine-Learning Ready Matrix:")
print(df_processed_features)

# Save the final matrix out to your structural JSON layer
df_processed_features.to_json('ml_ready_features.json', orient='records', indent=4)
print("\n💾 Success: Machine learning features exported to 'ml_ready_features.json'.")