import pandas as pd

def get_data():
    """
    Downloads the UCI Parkinson's dataset and standardizes it 
    for the pipeline architecture.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    
    # 'name' is a subject ID and has no predictive power
    # FIX: Remove ', axis=1' from this line
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
    
    # Standardize 'status' to 'target' so main.py doesn't break
    df = df.rename(columns={'status': 'target'})
    
    return df