import shap
def generate_shap_values(model, X_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    return explainer, shap_values