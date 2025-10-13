import pandas as pd
df_raw=pd.read_csv("data/raw/loan_approval_dataset.csv")
df_pro=pd.read_csv("data/processed/processed_loan_approval_dataset.csv")
print(df_pro.head())
print("--------------------------")
print(df_pro.columns)
# for col in (df.columns.tolist()):
    # print(col,df[col].max(),df[col].min())  



