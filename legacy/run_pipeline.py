import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def run():
    print("--- STARTING PIPELINE EXECUTION ---")
    file_path = 'PPMI_Curated_Data_Cut_Public_20260511.xlsx'
    
    # 1. Load Data
    try:
        df = pd.read_excel(file_path)
        print(f"Dataset loaded: {len(df)} rows.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Feature Selection
    features = [
        'age', 'SEX', 'EDUCYRS', 'duration',
        'DVT_TOTAL_RECALL', 'DVT_DELAYED_RECALL',
        'MIA_CAUDATE_BILAT', 'MIA_PUTAMEN_BILAT',
        'abeta', 'tau', 'ptau217_plasma',
        'upsit'
    ]
    target = 'updrs3_score'

    # 3. Data Preparation
    df_model = df[features + [target]].dropna(subset=[target])
    X = df_model[features]
    y = df_model[target]

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # 4. Hyperparameter Tuning
    print("Tuning hyperparameters (this may take a moment)...")
    param_grid = {
        'n_estimators': [200],
        'max_depth': [None],
        'min_samples_split': [2]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # 5. Evaluation
    preds = best_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n--- FINAL RESULTS ---")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 6. SHAP Interpretability
    print("\nGenerating SHAP explanation plot...")
    try:
        # Sample for speed to prevent hanging
        X_test_summary = shap.sample(X_test, 500)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_summary)
        
        shap.summary_plot(shap_values, X_test_summary, feature_names=features, show=False)
        plt.title("SHAP Feature Importance for UPDRS-III Prediction")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"SHAP generation skipped or failed: {e}")

if __name__ == "__main__":
    run()