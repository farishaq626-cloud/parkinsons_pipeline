import pandas as pd
def integrate_data(clinical_path, expression_path):
    clinical_df = pd.read_csv(clinical_path)
    expression_df = pd.read_csv(expression_path)
    merged_df = pd.merge(clinical_df, expression_df, on="patient_id", how="inner")
    return merged_df