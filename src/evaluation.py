import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

def evaluate_model(y_true, y_pred, model_name, y_probability=None):
    """
    Evaluate a model's performance using various metrics.
    
    Parameters:
    y_true: array-like
        True labels.
        
    y_pred: array-like
        Predicted labels.
        
    model_name: str
        Name of the model being evaluated.
        
    y_probability: array-like, optional
        Predicted probabilities for the positive class. Required for ROC AUC and log loss.
        
    Returns:
    dict
        Dictionary containing evaluation metrics.
    """
    
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred), 
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division = 0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    #ROC-AUC and log loss require predicted probabilities
    #calculate them only if they are provided
    
    if y_probability is not None:
        results["ROC-AUC"] = (roc_auc_score(y_true, y_probability))
        results["LogLoss"] = (log_loss(y_true, y_probability))
    
    #return one row to concatenate with other models' results later more easily
    return pd.DataFrame([results])