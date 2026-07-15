from ingest_engine import get_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_pipeline():
    # 1. Load Data
    df = get_data()
    
    # 2. Split Features and Target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Predictions and Evaluation
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("--- Pipeline Execution Results ---")
    print(classification_report(y_test, preds))
    print(f"AUC Score: {roc_auc_score(y_test, probs):.4f}")

if __name__ == "__main__":
    run_pipeline()