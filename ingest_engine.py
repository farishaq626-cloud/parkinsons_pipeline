import pandas as pd
import numpy as np
import json
from datetime import datetime

print("🔄 Parkinson's Progression Pipeline: Initializing Ingestion Layer...\n")

# 1. Simulating Unstructured, Messy Clinical Input Data
# In a real trial, this would be a messy CSV or Excel export with missing values and inconsistent formatting.
messy_clinical_data = {
    'Patient_ID': ['P-001', 'P-002', 'P-003', 'P-004'],
    'Age_At_Onset': [62, np.nan, 58, 71],  # Contains a missing value (NaN)
    'UPDRS_Part_III_Score': [24.5, 31.0, 18.5, np.nan],  # Motor assessment scores
    'Dopaminergic_Med': ['Levodopa', 'None', 'levodopa ', 'Agonist'],  # Inconsistent casing/spaces
    'Last_Visit_Date': ['2026-01-15', '12/05/2025', '2026-03-22', '02-11-2026']  # Messy date formats
}

# Convert raw input dictionary into a Pandas Dataframe
df_raw = pd.DataFrame(messy_clinical_data)
print("⚠️ Raw Input Data Snapshot:")
print(df_raw, "\n" + "-"*50 + "\n")


# 2. The Core Pipeline: Cleaning & Standardizing Layer
def clean_and_structure_pipeline(df):
    # Create a deep copy to preserve the original audit trail
    processed_df = df.copy()
    
    # Handle Missing Values: Impute missing numerical clinical features using column medians
    processed_df['Age_At_Onset'] = processed_df['Age_At_Onset'].fillna(processed_df['Age_At_Onset'].median())
    processed_df['UPDRS_Part_III_Score'] = processed_df['UPDRS_Part_III_Score'].fillna(processed_df['UPDRS_Part_III_Score'].median())
    
    # Standardize Categorical Data: Strip whitespaces and force lowercase uniform naming
    processed_df['Dopaminergic_Med'] = processed_df['Dopaminergic_Med'].str.strip().str.lower()
    
    return processed_df

# Run the cleaning engine
df_cleaned = clean_and_structure_pipeline(df_raw)


# 3. The Value Proposition: Exporting to Enterprise-Standard JSON
# Conglomerates buy assets that integrate seamlessly into cloud databases via JSON payloads.
structured_json_payload = df_cleaned.to_json(orient='records', indent=4)

print("🎯 Cleaned & Standardized Enterprise JSON Output:")
print(structured_json_payload)

# Save the structured asset locally
with open('structured_patient_data.json', 'w') as f:
    f.write(structured_json_payload)

print("\n💾 Success: Pipeline run complete. 'structured_patient_data.json' saved to workspace.")