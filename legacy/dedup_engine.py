def execute_deduplication(df):
    # Keep the last record per patient to ensure integrity
    return df.drop_duplicates(subset=['Patient_ID'], keep='last')