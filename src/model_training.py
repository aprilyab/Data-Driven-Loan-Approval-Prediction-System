import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from imblearn.over_sampling import SMOTE
from src.data_cleaning_and_preprocessing import data_cleaning



def model_training():
    scaler,df=data_cleaning()

    # split the dataset to predictor and target features
    x=df.drop(" loan_status",axis=1)
    y=df[" loan_status"]

    # split the dataset to train and split for training the model
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    smote=SMOTE(random_state=42)
    x_train,y_train=smote.fit_resample(x_train,y_train)
    
    # create model objects
    tree_model=DecisionTreeClassifier(criterion="entropy",max_depth=4)
    logistic_model=LogisticRegression()

    # model_training
    tree_model.fit(x_train,y_train)
    logistic_model.fit(x_train,y_train)

    # make prediction from models for evaluation
    logistic_y_pre=logistic_model.predict(x_test)
    tree_y_pre=tree_model.predict(x_test)

    # prediction probability for each class
    logistic_y_proba = logistic_model.predict_proba(x_test)
    tree_y_proba = tree_model.predict_proba(x_test)

    # plot the decision tree
    plot_tree(tree_model,feature_names=x_train.columns, 
          class_names=["Rejected", "Approved"],  # class names
          filled=True,  # colors nodes by class
          rounded=True,  # rounded boxes
          fontsize=12)   # font size)
    plt.savefig("outputs/figures/Decision_Tree_Model_plot")
    plt.show()
    

    # save the models
    joblib.dump(logistic_model,"outputs/models/Logical_Regression_Loan_Approval_Predictor_model.pkl")
    joblib.dump(tree_model,"outputs/models/Decision_Tree_Loan_Approval_Predictor_model.pkl")
    joblib.dump(scaler,"outputs/Scaler")

    return y_test,tree_y_pre,logistic_y_pre,tree_y_proba,logistic_y_proba

if __name__=="__main__":
    model_training()