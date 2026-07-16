import pandas as pd
import json

class DataLoader:
    def __init__(self, mode="UCI"):
        self.mode = mode
        with open('config.json', 'r') as f:
            self.config = json.load(f)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        if self.mode == "PPMI":
            return self._transform_ppmi(df)
        return df

    def _transform_ppmi(self, df):
        # This fixes the APPRDX to COHORT requirement you found
        if 'APPRDX' in df.columns:
            df = df.rename(columns={'APPRDX': 'COHORT'})
        return df