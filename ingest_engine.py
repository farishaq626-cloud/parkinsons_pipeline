import pandas as pd

def clean_and_structure_pipeline(df):
    # Ensure numerical types for critical columns
    df['Age_At_Onset'] = pd.to_numeric(df['Age_At_Onset'], errors='coerce')
    df['UPDRS_Motor_Score'] = pd.to_numeric(df['UPDRS_Motor_Score'], errors='coerce')
    # Drop rows with critical missing data
    return df.dropna(subset=['Age_At_Onset', 'UPDRS_Motor_Score', 'Patient_ID'])