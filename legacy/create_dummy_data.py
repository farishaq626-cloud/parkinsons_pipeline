import pandas as pd
import os
import numpy as np

def create_dummy_data():
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' folder.")
    
    # Create mock clinical data
    clinical_df = pd.DataFrame({
        'patient_id': range(1, 11),
        'diagnosis': np.random.choice([0, 1], size=10)
    })
    clinical_df.to_csv('data/clinical.csv', index=False)
    
    # Create mock expression data
    expression_df = pd.DataFrame({
        'patient_id': range(1, 11),
        'gene_A': np.random.rand(10),
        'gene_B': np.random.rand(10)
    })
    expression_df.to_csv('data/expression.csv', index=False)
    
    print("Files 'clinical.csv' and 'expression.csv' created successfully in the 'data' folder.")

if __name__ == "__main__":
    create_dummy_data()