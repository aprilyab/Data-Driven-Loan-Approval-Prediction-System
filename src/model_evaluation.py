from src.model_training import model_training
from sklearn.metrics import accuracy_score,log_loss
import json
import joblib


def evaluation():
    logistic_model=joblib.load("outputs/models/Logical_Regression_Loan_Approval_Predictor_model.pkl")
    scaler=joblib.load("outputs/Scaler")
    tree_mode=joblib.load("outputs/models/Decision_Tree_Loan_Approval_Predictor_model.pkl")
    
    # load files for evaluation
    y_test,tree_y_pre,logistic_y_pre,tree_y_proba,logistic_y_proba=model_training()
    
    # evaluation vie accuracy score
    as_tree=accuracy_score(y_test,tree_y_pre)
    as_logistic=accuracy_score(y_test,logistic_y_pre)
    
    # evaluation via log loss
    ll_tree=log_loss(y_test,tree_y_proba)
    ll_logistic=log_loss(y_test,logistic_y_proba)
    
    # create metrics dictionary
    metrics = {
        "Decision Tree": {
            "Accuracy Score": as_tree,
            "Log Loss": ll_tree
        },
        "Logistic Regression": {
            "Accuracy Score": as_logistic,
            "Log Loss": ll_logistic
        }
    }

    # save the evaluation matrics
    with open ("outputs/metrics.json","w") as f:
        json.dump(metrics,f)

if __name__=="__main__":
    evaluation()
