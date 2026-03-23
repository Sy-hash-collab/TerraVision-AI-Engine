import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def print_ffc_business_metrics(y_true, y_pred_probs, model_name="Model", threshold=0.5):
    """
    Answers Q6 & Q9 Tasks: Evaluation metrics with a business focus for FFC.
    """
    preds = (np.array(y_pred_probs) > threshold).astype(int).flatten()
    
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    
    print(f"--- FFC Executive Summary: {model_name} ---")
    print(f"Accuracy:  {acc:.4f} (Overall Correctness)")
    print(f"Precision: {prec:.4f} (When we predict Agri, how often are we right?)")
    print(f"Recall:    {rec:.4f} (Out of all Agri land, how much did we find?)")
    print(f"F1-Score:  {f1:.4f} (Harmonic mean - Key metric for imbalanced boundaries)")
    print("Confusion Matrix:")
    print(cm)
    print("False Negatives (Missed Agricultural Land):", cm[1][0])
    print("-" * 50)

if __name__ == "__main__":
    # Dummy test
    print_ffc_business_metrics([1, 0, 1, 1, 0], [0.9, 0.1, 0.4, 0.8, 0.2], model_name="Test Model")
