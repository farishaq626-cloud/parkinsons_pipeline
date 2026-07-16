import pandas as pd
import os
import glob

# Search for any file with PPMI in the name
files = glob.glob("*PPMI*.csv") + glob.glob("*PPMI*.xlsx")

if not files:
    print("Error: Could not find any file with 'PPMI' in the name.")
else:
    file_path = files[0]
    print(f"Loading: {file_path}")
    
    # Load data
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Print the columns
    print("\n--- Exact Column Names found in file ---")
    for col in df.columns:
        print(col)