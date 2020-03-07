import sklearn
import pandas as pd 
from sklearn.preprocessing import Binarizer
dataset_raw = pd.read_csv("Datasets\\telco-customer-churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
#print(dataset_raw.head())


#print(dataset_raw.isna().sum())

#No data imputation is needed
#print(dataset_raw.OnlineBackup.unique())
print(dataset_raw.columns.values)