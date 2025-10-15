from src.model_training import model_training
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
import json
import joblib

def evaluation():
    # Load trained models
    logistic_model = joblib.load("outputs/models/Logical_Regression_Loan_Approval_Predictor_model.pkl")
    tree_model = joblib.load("outputs/models/Decision_Tree_Loan_Approval_Predictor_model.pkl")
    scaler = joblib.load("outputs/Scaler")
    
    # Load data for evaluation
    y_test, tree_y_pred, logistic_y_pred, tree_y_proba, logistic_y_proba = model_training()
    
    # --------------------
    # Accuracy
    # --------------------
    acc_tree = accuracy_score(y_test, tree_y_pred)
    acc_logistic = accuracy_score(y_test, logistic_y_pred)
    
    # --------------------
    # Log Loss
    # --------------------
    ll_tree = log_loss(y_test, tree_y_proba)
    ll_logistic = log_loss(y_test, logistic_y_proba)
    
    # --------------------
    # Precision, Recall, F1-score
    # --------------------
    precision_tree = precision_score(y_test, tree_y_pred)
    recall_tree = recall_score(y_test, tree_y_pred)
    f1_tree = f1_score(y_test, tree_y_pred)
    
    precision_logistic = precision_score(y_test, logistic_y_pred)
    recall_logistic = recall_score(y_test, logistic_y_pred)
    f1_logistic = f1_score(y_test, logistic_y_pred)
    
    # --------------------
    # Combine metrics
    # --------------------
    metrics = {
        "Decision Tree": {
            "Accuracy Score": acc_tree,
            "Log Loss": ll_tree,
            "Precision": precision_tree,
            "Recall": recall_tree,
            "F1 Score": f1_tree
        },
        "Logistic Regression": {
            "Accuracy Score": acc_logistic,
            "Log Loss": ll_logistic,
            "Precision": precision_logistic,
            "Recall": recall_logistic,
            "F1 Score": f1_logistic
        }
    }
    
    # Save metrics to JSON
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Evaluation completed. Metrics saved to outputs/metrics.json")

if __name__ == "__main__":
    evaluation()
