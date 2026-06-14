import pandas as pd

def validate_clinical_data(df):
    """
    Checks if the incoming data meets the enterprise clinical standard.
    Includes defensive checks to prevent runtime crashes.
    """
    # 0. Empty Dataframe Check
    if df.empty:
        return False, "CRITICAL ERROR: Input dataframe is empty."

    required_columns = ['Patient_ID', 'Visit_Month', 'UPDRS_Part_III_Score']

    # 1. Schema Check
    for col in required_columns:
        if col not in df.columns:
            return False, f"CRITICAL ERROR: Missing required column: {col}"

    # 2. Type Check (Defensive Coercion)
    # Convert column to numeric, forcing errors (like strings) to NaN
    coerced_scores = pd.to_numeric(df['UPDRS_Part_III_Score'], errors='coerce')

    # If any value is NaN, it means it couldn't be converted to a number
    if coerced_scores.isna().any():
        return False, "CRITICAL ERROR: Non-numeric data detected in UPDRS_Part_III_Score."

    # 3. Value Range Check
    # Now safe to compare because we've verified they are all numeric
    if (coerced_scores < 0).any():
        return False, "CRITICAL ERROR: Invalid clinical data detected (Negative UPDRS score)"

    return True, "Data validation passed."