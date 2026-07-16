import pandas as pd
import os

# Define the file name
file_name = 'ml_ready_features.json'

# Verify file exists before loading
if os.path.exists(file_name):
    # Load the file
    df = pd.read_json(file_name)
    
    # Print the columns
    print("--- Exact Column Names found in file ---")
    print(df.columns.tolist())
else:
    print(f"Error: Could not find {file_name}. Check if you are in the correct directory.")