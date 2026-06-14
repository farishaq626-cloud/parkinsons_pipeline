import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

print("📊 Parkinson's Pipeline: Initializing Advanced Visual Diagnostics Engine...\n")

def generate_pipeline_diagnostics(y_true, y_pred, y_probs):
    """
    Generates professional validation plots for the machine learning layer:
    1. A Text-based Confusion Matrix
    2. A saved ROC-AUC Curve plot file
    """
    # 1. Compute the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("📋 True Performance Breakdown:")
    print(f"   • True Negatives  (Correctly identified Stable): {tn}")
    print(f"   • False Positives (Stable misclassified as Rapid): {fp}")
    print(f"   • False Negatives (Rapid missed by model): {fn}")
    print(f"   • True Positives  (Correctly caught Rapid): {tp}\n")
    
    # 2. Compute the ROC-AUC Score
    auc_score = roc_auc_score(y_true, y_probs)
    print(f"✨ Computed Pipeline ROC-AUC Score: {auc_score:.4f}")
    
    # 3. Plotting the ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing Baseline')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Clinical Classifier Validation: ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the output visualization
    output_filename = 'pipeline_roc_curve.png'
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"💾 Production Asset Saved successfully: '{output_filename}' generated in your workspace.")

if __name__ == "__main__":
    # Test run using values from your main pipeline output
    mock_true =  np.array([0, 1, 1, 0, 0, 0, 0])
    mock_pred =  np.array([0, 1, 1, 0, 0, 0, 0])
    mock_probs = np.array([0.15, 0.88, 0.92, 0.34, 0.12, 0.05, 0.21])
    generate_pipeline_diagnostics(mock_true, mock_pred, mock_probs)