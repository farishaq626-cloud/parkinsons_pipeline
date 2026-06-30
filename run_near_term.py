# run_near_term.py
# This script orchestrates your near-term pipeline modules

from near_term_enhancements.gene_integration import integrate_data
from near_term_enhancements.biomarker_discovery import get_top_biomarkers
from near_term_enhancements.xai_engine import generate_shap_values
from near_term_enhancements.automated_reporting import save_report

def main():
    try:
        print("Starting pipeline execution...")
        
        # 1. Integration
        # Ensure 'clinical.csv' and 'expression.csv' are in the 'data' folder
        df = integrate_data("data/clinical.csv", "data/expression.csv")
        print("Data integration successful.")
        
        # 2. Biomarker Discovery
        biomarkers, model = get_top_biomarkers(df, target="diagnosis")
        print("Biomarker discovery complete.")
        
        # 3. XAI
        X = df.drop(columns=["patient_id", "diagnosis"])
        explainer, shap_vals = generate_shap_values(model, X)
        print("SHAP values generated.")
        
        # 4. Save
        save_report(biomarkers.to_dict())
        print("SUCCESS: Analysis complete. Check report.json for output.")

    except FileNotFoundError:
        print("ERROR: Could not find data files. Check if they are in the 'data' folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()