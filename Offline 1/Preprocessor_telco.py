import sklearn
import pandas as pd 
from sklearn.preprocessing import Binarizer

        
dataset_raw = pd.read_csv("Datasets\\telco-customer-churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
#print(dataset_raw.head())


print(dataset_raw.isnull().sum())

#No data imputation is needed
#print(dataset_raw.OnlineBackup.unique())
#print(dataset_raw.columns.values)
Attributes = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
 'StreamingMovies', 'Contract', 'PaperlessBilling' ,'PaymentMethod','MonthlyCharges', 'TotalCharges']
Label =['Churn']


unique_vals_tenure = list(dataset_raw.tenure.unique())
#print(unique_vals_tenure)
unique_vals_tenure.sort()
#print(unique_vals_tenure)
mean = sum(unique_vals_tenure)/len(unique_vals_tenure)
print(mean)
dataset_raw['tenure'] = (dataset_raw['tenure'] > mean).astype(bool)
#print(dataset_raw.head())

vals_monthly_charges = list(dataset_raw.MonthlyCharges)
#print(unique_vals_monthly_charges)
mean_1 = sum(vals_monthly_charges) /len(vals_monthly_charges)
print(mean_1)
dataset_raw['MonthlyCharges'] = (dataset_raw['MonthlyCharges'] > mean_1).astype(bool)

vals_Total_charges = list(dataset_raw.TotalCharges)
print(len(vals_Total_charges))

sum = 0
count = 0
for i in range(len(vals_Total_charges)):
    #print(i)
    if vals_Total_charges[i]!=' ':
        
       # print("yes")
        vals_Total_charges[i]=float(vals_Total_charges[i])
        sum = sum+vals_Total_charges[i]
        count = count + 1
mean_to_impute = sum/count

for i in range(len(vals_Total_charges)):
    #print(i)
    if vals_Total_charges[i]==' ':
        
        #print("yes")
        vals_Total_charges[i]=mean_to_impute
        sum = sum+vals_Total_charges[i]
        count = count + 1  

threshold_mean = sum/count
dataset_raw['TotalCharges'] = vals_Total_charges
dataset_raw['TotalCharges'] = (dataset_raw['TotalCharges'] > threshold_mean).astype(bool)
dataset_raw.to_csv("test_Telco.csv")