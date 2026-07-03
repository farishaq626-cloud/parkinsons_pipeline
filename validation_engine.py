def validate_clinical_data(df):
    # Filter out invalid negative scores to prevent crashes
    return df[df['UPDRS_Motor_Score'] >= 0]