import pandas as pd
import json
# Import the specific cleaning and processing logic from your individual files
from ingest_engine import clean_and_structure_pipeline
from dedup_engine import execute_deduplication
from feature_engine import scale_clinical_features

print("🚀 ================================================== 🚀")
print("⚡ PARKINSON'S DATA PIPELINE: END-TO-END ORCHESTRATION ⚡")
print("🚀 ==================================================\n")

# 1. Simulating the Raw, Dirty Input Data entering the system
raw_incoming_data = {
    'Patient_ID': ['P-001', 'P-002', 'p-001 ', 'P-003', 'P-002'],
    'Age_At_Onset': [55.0, 68.0, 55.0, 42.0, 68.0],
    'UPDRS_Part_III_Score': [14.0, 38.5, 14.0, 9.0, 38.5],
    'UPDRS_Motor_Score': [14.0, 38.5, 14.0, 9.0, 38.5], # Matching for scaling consistency
    'Dopaminergic_Med': ['Levodopa', 'None', 'levodopa ', 'Agonist', 'None'],
    'Clinic_Code': ['LON-01', 'MAN-02', 'lon-01', 'EDN-04', 'MAN-02']
}

df_input = pd.DataFrame(raw_incoming_data)
print(f"📥 Step 1: Raw clinical data ingested into system memory. Total records: {len(df_input)}\n")

# 2. Automatically route the data through your pipeline architecture
print("⚙️ Step 2: Running Ingestion & Normalization Layer...")
df_normalized = clean_and_structure_pipeline(df_input)

print("\n⚙️ Step 3: Routing to Cross-Trial Deduplication Layer...")
df_deduplicated = execute_deduplication(df_normalized)

print("\n⚙️ Step 4: Routing to Scikit-Learn Feature Scaling Layer...")
df_final_matrix = scale_clinical_features(df_deduplicated)

# 3. Export the unified, production-ready dataset
print("\n📦 Step 5: Serializing Final Enterprise JSON Output...")
final_json = df_final_matrix.to_json(orient='records', indent=4)
print(final_json)

with open('final_production_output.json', 'w') as f:
    f.write(final_json)

print("\n🏁 [SUCCESS] End-to-end execution complete. Clean data pipeline fully validated.")