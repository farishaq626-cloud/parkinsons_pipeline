import pandas as pd
import glob

# 1. Automatically locate the PPMI dataset
files = glob.glob("*PPMI*.csv") + glob.glob("*PPMI*.xlsx")

if not files:
    print("ERROR: No PPMI file found in this directory.")
else:
    file_path = files[0]
    print(f"Reading file: {file_path}")
    
    # 2. Load just the headers to save time
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, nrows=0)
    else:
        df = pd.read_excel(file_path, nrows=0)
    
    # 3. Display the features
    print("\n--- AVAILABLE CLINICAL FEATURES ---")
    for col in df.columns:
        print(col)
    print("-----------------------------------")