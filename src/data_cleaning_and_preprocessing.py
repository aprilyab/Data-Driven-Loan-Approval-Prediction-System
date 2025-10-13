import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def data_cleaning():
            
            # load the raw data
            df=pd.read_csv("data/raw/loan_approval_dataset.csv")
            
            # drop missing values
            df.dropna(inplace=True)

            # count the number of missing values
            missing=df.isnull().sum()

            # drop load_id
            df.drop("loan_id",axis=1,inplace=True)

            # count the number of the duplicated rows
            duplicated=df.duplicated().sum()

            cat_cols=df.select_dtypes(include=["object","category"]).columns.tolist()
            num_cols=df.select_dtypes(include=["float64","int64"]).columns.tolist()

            # scale the numerical columns
            scaler=StandardScaler()
            scaled_df=scaler.fit_transform(df[num_cols])

            scaled_df=pd.DataFrame(scaled_df,columns=scaler.get_feature_names_out(num_cols))

            # merge with the origin data
            df=pd.concat([df.drop(columns=num_cols),scaled_df],axis=True)

            # encode the category columns
            # remove the targe column
            cat_cols.remove(" loan_status")

            encoder=OneHotEncoder(sparse_output=False,drop="first")

            encoded_df=encoder.fit_transform(df[cat_cols])

            # change category data to datadrame
            encoded_df=pd.DataFrame(encoded_df,columns=encoder.get_feature_names_out(cat_cols))

            # encode the target feature
            target_encoder=LabelEncoder()
            df[" loan_status"]=target_encoder.fit_transform(df[" loan_status"])

            # invert (0->1,1->0)
            df[" loan_status"]=df[" loan_status"].apply(lambda x: 1-x)
            # marge with the main data frame
            df=pd.concat([df.drop(columns=cat_cols),encoded_df],axis=1)

            df.to_csv("data/processed/processed_loan_approval_dataset.csv",index=False)
    
            return scaler,df

if __name__=="__main__":
        data_cleaning()



                