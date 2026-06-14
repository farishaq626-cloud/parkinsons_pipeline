import pandas as pd
import json

print("🧠 Parkinson's Pipeline: Initializing Clinical Deduplication Engine...\n")

# 1. Simulating Duplicated Cross-Trial Clinical Inputs
# Notice P-001 is duplicated with conflicting cases and trailing spaces.
cross_trial_data = {
    'Patient_ID': ['P-001', 'P-002', 'p-001 ', 'P-003', 'P-002'],
    'Clinic_Code': ['LON-01', 'MAN-02', 'lon-01', 'EDN-04', 'MAN-02'],
    'UPDRS_Baseline': [24.5, 31.0, 24.5, 18.5, 31.0]
}

df_dirty = pd.DataFrame(cross_trial_data)
print("⚠️ Raw Consolidated Datasets (Before Deduplication):")
print(df_dirty)
print(f"Total starting records: {len(df_dirty)}\n" + "-"*50 + "\n")


# 2. Automated Token Cleansing & Deduplication Logic
def execute_deduplication(df):
    initial_count = len(df)
    
    # Standardize string keys to catch hidden duplicates (lowercase & remove whitespace)
    df['Patient_ID_Clean'] = df['Patient_ID'].astype(str).str.strip().str.upper()
    df['Clinic_Code_Clean'] = df['Clinic_Code'].astype(str).str.strip().str.upper()
    
    # Drop rows where the unique combination of Patient and Clinic already exists
    # 'keep=first' retains the original entry and flags the subsequent ones as duplicates
    df_cleaned = df.drop_duplicates(subset=['Patient_ID_Clean', 'Clinic_Code_Clean'], keep='first')
    
    # Drop the temporary processing columns before outputting
    df_final = df_cleaned.drop(columns=['Patient_ID_Clean', 'Clinic_Code_Clean'])
    
    final_count = len(df_final)
    print(f"📉 Deduplication Complete: Removed {initial_count - final_count} duplicate records.")
    
    return df_final

# Run the engine
df_deduplicated = execute_deduplication(df_dirty)


# 3. Output the Golden Dataset
print("\n🎯 Cleaned Golden Dataset (No Data Leakage):")
print(df_deduplicated)

# Export the clean audit record
df_deduplicated.to_json('deduplicated_patient_registry.json', orient='records', indent=4)
print("\n💾 Success: Deduplication complete. 'deduplicated_patient_registry.json' saved.")