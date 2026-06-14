import pandas as pd
import numpy as np
from validation_engine import validate_clinical_data

print("🧪 Running Pipeline Stress Test: Ensuring Robustness against clinical data errors...")

def run_stress_test():
    # 1. Create a "Poisoned" Dataset (Simulating real-world entry errors)
    # Error: Missing Age_At_Onset (NaN) and incorrect Data Types
    poisoned_data = {
        'Patient_ID': ['P-999', 'P-999'],
        'Visit_Month': [0, 12],
        'Age_At_Onset': [np.nan, 55.0],  # Critical Error: Missing Value
        'UPDRS_Part_III_Score': [15.0, "Twenty-Two"], # Critical Error: String where float expected
        'Clinic_Code': ['LON-01', 'LON-01'],
        'Dopaminergic_Med': ['None', 'None']
    }
    
    df_poisoned = pd.DataFrame(poisoned_data)
    
    print("\n[TEST 1] Testing missing/corrupted data detection...")
    is_valid, message = validate_clinical_data(df_poisoned)
    
    if not is_valid:
        print(f"✅ Success: Validation Gatekeeper correctly halted pipeline.")
        print(f"   Reason captured: {message}")
    else:
        print(f"❌ Failed: The pipeline accepted bad data!")

    # 2. Test empty dataframe
    print("\n[TEST 2] Testing empty dataset detection...")
    df_empty = pd.DataFrame()
    is_valid, message = validate_clinical_data(df_empty)
    
    if not is_valid:
        print(f"✅ Success: Validation Gatekeeper correctly identified empty input.")
    else:
        print(f"❌ Failed: The pipeline accepted an empty dataframe!")

if __name__ == "__main__":
    run_stress_test()