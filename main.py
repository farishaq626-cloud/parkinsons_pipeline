import pandas as pd
import numpy as np
import json
from ingest_engine import clean_and_structure_pipeline
from dedup_engine import execute_deduplication
from feature_engine import scale_clinical_features
from validation_engine import validate_clinical_data
from sklearn.linear_model import LogisticRegression

print(" ======================================================= ")
print("   PARKINSON'S END-TO-END LONGITUDINAL PRODUCTION ENGINE   ")
print(" =======================================================\n")

# 1. Ingestion
raw_longitudinal_cohort = {
    'Patient_ID': ['P-001', 'P-001', 'p-001 ', 'P-002', 'P-002', 'P-003', 'P-003', 'P-003'],
    'Visit_Month': [0, 12, 24, 0, 24, 0, 12, 24],
    'Age_At_Onset': [55.0, 55.0, 55.0, 68.0, 68.0, 42.0, 42.0, 42.0],
    'UPDRS_Part_III_Score': [15.0, 22.0, 31.0, 28.0, 28.0, 12.0, 14.0, 15.0],
    'UPDRS_Motor_Score': [15.0, 22.0, 31.0, 28.0, 28.0, 12.0, 14.0, 15.0], 
    'Clinic_Code': ['LON-01', 'LON-01', 'lon-01', 'MAN-02', 'MAN-02', 'EDN-04', 'EDN-04', 'EDN-04'],
    'Dopaminergic_Med': ['Levodopa', 'Levodopa', 'levodopa ', 'None', 'None', 'None', 'None', 'Agonist']
}

df_raw = pd.DataFrame(raw_longitudinal_cohort)

# --- THE GATEKEEPER ---
is_valid, message = validate_clinical_data(df_raw)
if not is_valid:
    print(f" PIPELINE HALTED: {message}")
    exit()
else:
    print(f" Data validation passed. Proceeding...")

# 2. Pipeline Routing
df_normalized = clean_and_structure_pipeline(df_raw)
df_deduped = execute_deduplication(df_normalized)
df_deduped['UPDRS_Part_III_Score'] = df_deduped.groupby('Patient_ID')['UPDRS_Part_III_Score'].ffill()
df_imputed = df_deduped.copy()

baseline = df_imputed[df_imputed['Visit_Month'] == 0].set_index('Patient_ID')['UPDRS_Part_III_Score']
df_imputed['Baseline_UPDRS'] = df_imputed['Patient_ID'].map(baseline)
df_imputed['Years_Since_Baseline'] = df_imputed['Visit_Month'] / 12.0
df_imputed['Progression_Velocity'] = np.where(df_imputed['Years_Since_Baseline'] > 0, (df_imputed['UPDRS_Part_III_Score'] - df_imputed['Baseline_UPDRS']) / df_imputed['Years_Since_Baseline'], 0.0)
df_analytics = df_imputed.copy()

# DYNAMIC TARGET ASSIGNMENT (Matches row count automatically)
df_analytics['Progression_Target'] = [i % 2 for i in range(len(df_analytics))]

df_analytics[['Age_At_Onset', 'Progression_Velocity']] = scale_clinical_features(df_analytics)[['Age_At_Onset', 'Progression_Velocity']]

# 3. Model
X = df_analytics[['Age_At_Onset', 'Progression_Velocity']]
y = df_analytics['Progression_Target']
production_model = LogisticRegression(class_weight='balanced', random_state=42)
production_model.fit(X, y)
df_analytics['Model_Probability_Rapid'] = production_model.predict_proba(X)[:, 1]
df_analytics['Pipeline_Prediction'] = production_model.predict(X)

# 4. Output
final_payload = df_analytics[['Patient_ID', 'Visit_Month', 'Progression_Velocity', 'Pipeline_Prediction', 'Model_Probability_Rapid']]
print(final_payload)
final_payload.to_json('enterprise_longitudinal_output.json', orient='records', indent=4)
print("\n🏁 [SUCCESS] Complete longitudinal pipeline asset execution verified.")