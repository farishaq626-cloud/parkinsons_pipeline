import pandas as pd
import numpy as np

print(" Parkinson's Pipeline: Initializing Longitudinal Analytics Engine...\n")

# 1. Simulating a Multi-Visit Clinical Trial Dataset (e.g., PPMI style)
# Patients are tracked over multiple years. Notice some missing values (NaN) 
# which naturally happens when patients miss a clinic appointment.
longitudinal_data = {
    'Patient_ID': ['P-001', 'P-001', 'P-001', 'P-002', 'P-002', 'P-003', 'P-003', 'P-003'],
    'Visit_Month': [0, 12, 24, 0, 24, 0, 12, 24],  # Timeline in months
    'UPDRS_Total': [15.0, 22.0, 31.0, 28.0, np.nan, 12.0, 14.0, 15.0], # Clinical motor scores
    'LEDD_Dose':   [300, 400, 600, 450, 450, 0, 0, 150]    # Levodopa Equivalent Daily Dose (mg)
}

df_long = pd.DataFrame(longitudinal_data)
print(" Raw Longitudinal Cohort Records Ingested:")
print(df_long)
print("-" * 60 + "\n")


# 2. Imputation Layer: Handling Clinical Attrition (Forward-Fill)
def impute_longitudinal_data(df):
    """
    In clinical tracking, if a patient misses a visit, we safely forward-fill 
    their previous known score to ensure the pipeline doesn't break.
    """
    df_imputed = df.copy()
    
    # Group by patient so we don't accidentally fill data using a different patient's scores
    df_imputed['UPDRS_Total'] = df_imputed.groupby('Patient_ID')['UPDRS_Total'].ffill()
    
    print(" Attrition Mitigated: Forward-fill imputation applied to missing visit metrics.")
    return df_imputed


# 3. Feature Engineering: Calculating the Progression Velocity Vector
def calculate_progression_velocity(df):
    """
    Computes the exact annual rate of change for each patient's UPDRS score.
    Velocity = (Current_Score - Baseline_Score) / Time_Delta_In_Years
    """
    df_velocity = df.copy()
    
    # Identify baseline metrics (Visit_Month == 0) for each patient
    baseline = df_velocity[df_velocity['Visit_Month'] == 0].set_index('Patient_ID')['UPDRS_Total']
    
    # Map the baseline values back to every row corresponding to that patient
    df_velocity['Baseline_UPDRS'] = df_velocity['Patient_ID'].map(baseline)
    
    # Calculate the time delta in years
    df_velocity['Years_Since_Baseline'] = df_velocity['Visit_Month'] / 12.0
    
    # Calculate velocity (avoiding division by zero at baseline visit)
    df_velocity['Progression_Velocity'] = np.where(
        df_velocity['Years_Since_Baseline'] > 0,
        (df_velocity['UPDRS_Total'] - df_velocity['Baseline_UPDRS']) / df_velocity['Years_Since_Baseline'],
        0.0  # Velocity is naturally zero at the baseline start point
    )
    
    print(" Feature Vector Engineered: Calculated annual Progression Velocity slopes.")
    return df_velocity

# Execute the engine steps locally for validation
df_cleaned_long = impute_longitudinal_data(df_long)
df_analyzed = calculate_progression_velocity(df_cleaned_long)

print("\n Advanced Longitudinal Analysis Output Matrix:")
print(df_analyzed[['Patient_ID', 'Visit_Month', 'UPDRS_Total', 'Progression_Velocity', 'LEDD_Dose']])