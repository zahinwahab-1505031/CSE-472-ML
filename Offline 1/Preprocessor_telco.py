import sklearn
import pandas as pd 
from sklearn.preprocessing import Binarizer
import math
        
dataset_raw = pd.read_csv("Datasets\\telco-customer-churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
def calculate_entropy_step2(q):
    B1 = 0
    if q == 0:
        B1=0
    else:
        B1 = -q*math.log(q,2) 
    #print(B1)
    #print(1-q)
    B2=0
    if (1-q)==0:
        B2=0
    else:
        B2 = -(1-q)*math.log(1-q,2)
    B = B1+B2
    return B


    


def binarize_using_infogain(Attribute_to_binarize,dataset_raw,label):
    values_to_binarize = list(dataset_raw[Attribute_to_binarize].unique())
    values_to_binarize.sort()

    print(len(values_to_binarize))

    split_points = []
    res = (values_to_binarize[0]-abs((values_to_binarize[0]-values_to_binarize[1])/2))
    split_points.append(res)
    for iter in range(len(values_to_binarize)-1):
        res = ((values_to_binarize[iter]+values_to_binarize[iter+1])/2)
        split_points.append(res)
    res = (values_to_binarize[len(values_to_binarize)-1]+((values_to_binarize[len(values_to_binarize)-1]-values_to_binarize[len(values_to_binarize)-2])/2))
    split_points.append(res)
    print("=================================================================================")
    #print(split_points[len(split_points)-10:])
    print(len(split_points))
    max_ig = -99999
    split_points = list(set(split_points))
    total_samples = dataset_raw.shape[0] 
    #split_points = np.array(split_points) 
    #split_points = np.unique(split_points)
    
    split_points.sort()
    print(split_points)
    final_split = split_points[0]
    total_samples_positive= len(dataset_raw[dataset_raw[label]=='Yes'])
    ParentEntropy = calculate_entropy_step2(total_samples_positive/total_samples)
    for split in split_points:
        group_1 = dataset_raw[dataset_raw[Attribute_to_binarize] <split]
        group_1_class_1 = group_1[group_1[label]=='Yes']
        group_1_class_0 = group_1[group_1[label]=='No']

        pk = group_1_class_1.shape[0]
        nk = group_1_class_0.shape[0]
        if (pk+nk)==0:
            childentropy1 = 0
        else:
            childentropy1 = ((pk+nk)/total_samples)*calculate_entropy_step2(pk/(pk+nk))
      #  print("CHECKPOINT 1")

        #print(group_1.shape[0],group_1_class_1.shape[0],group_1_class_0.shape[0])
       
        
        group_2 = dataset_raw[dataset_raw[Attribute_to_binarize] >=split]
        group_2_class_1 = group_2[group_2[label]=='Yes']
        group_2_class_0 = group_2[group_2[label]=='No']

        pk = group_2_class_1.shape[0]
        nk = group_2_class_0.shape[0]
        if (pk+nk)==0:
            childentropy2 = 0
        else:
            childentropy2 = ((pk+nk)/total_samples)*calculate_entropy_step2(pk/(pk+nk))
        
        ig = ParentEntropy - childentropy1 - childentropy2
        
        if max_ig < ig:
            max_ig = ig
            final_split = split
    print("FINAL SPLIT: ",final_split)
    return final_split
def binarize_using_gini(Attribute_to_binarize,dataset_raw,label):
    values_to_binarize = list(dataset_raw[Attribute_to_binarize].unique())
    values_to_binarize.sort()
    print(len(values_to_binarize))
    #print((values_to_binarize))

    split_points = []
    res = math.floor(values_to_binarize[0]-abs((values_to_binarize[0]-values_to_binarize[1])/2))
    split_points.append(res)
    for iter in range(len(values_to_binarize)-1):
        res = math.floor((values_to_binarize[iter]+values_to_binarize[iter+1])/2)
        split_points.append(res)
    res = math.floor(values_to_binarize[len(values_to_binarize)-1]+abs((values_to_binarize[len(values_to_binarize)-1]-values_to_binarize[len(values_to_binarize)-2])/2))
    split_points.append(res)
    print(len(split_points))
    #print((split_points))
    min_gini = 99999
    final_split = split_points[0]
    total_samples = dataset_raw.shape[0]  
    for split in split_points:
        #print(split)
        group_1 = dataset_raw[dataset_raw[Attribute_to_binarize] <split]
        group_1_class_1 = group_1[group_1[label]=='Yes']
        group_1_class_0 = group_1[group_1[label]=='No']

        #print(group_1.shape[0],group_1_class_1.shape[0],group_1_class_0.shape[0])

        group_2 = dataset_raw[dataset_raw[Attribute_to_binarize] >=split]
        group_2_class_1 = group_2[group_2[label]=='Yes']
        group_2_class_0 = group_2[group_2[label]=='No']

        #print(group_2.shape[0],group_2_class_1.shape[0],group_2_class_0.shape[0])
        #print(dataset_raw.shape[0],group_1.shape[0],group_2.shape[0])
        if group_1.shape[0]==0:
            group_1_class_0_prop = 0
        else:
            group_1_class_0_prop = group_1_class_0.shape[0] / group_1.shape[0]
        if group_1.shape[0]==0:
            group_1_class_1_prop = 0
        else:
            group_1_class_1_prop = group_1_class_1.shape[0] / group_1.shape[0]
        if group_2.shape[0]==0:
            group_2_class_0_prop = 0
        else:
            group_2_class_0_prop = group_2_class_0.shape[0] / group_2.shape[0]
        if group_2.shape[0]==0:
            group_2_class_1_prop = 0
        else:
            group_2_class_1_prop = group_2_class_1.shape[0] / group_2.shape[0]

        #gini_index = sum(proportion * (1.0 - proportion))

        gini_index_group_1 = (1.0 - (group_1_class_0_prop * group_1_class_0_prop)-(group_1_class_1_prop*group_1_class_1_prop)) * (group_1.shape[0]/total_samples)
            

        gini_index_group_2 = (1.0 - (group_2_class_0_prop * group_2_class_0_prop)-(group_2_class_1_prop*group_2_class_1_prop)) * (group_2.shape[0]/total_samples)
        gini_index = gini_index_group_1+gini_index_group_2
        #print(gini_index)
        if gini_index < min_gini:
            min_gini = gini_index
            final_split = split  
            #print("DO")
    print("FINAL SPLIT: ",final_split)
    return final_split


#print(dataset_raw.isnull().sum())
Attributes = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
 'StreamingMovies', 'Contract', 'PaperlessBilling' ,'PaymentMethod','MonthlyCharges', 'TotalCharges']
label ='Churn'


split = binarize_using_infogain('tenure',dataset_raw,label)

#dataset_raw['tenure'] = (dataset_raw['tenure'] > split).astype(bool)

#dataset_raw['MonthlyCharges'] = (dataset_raw['MonthlyCharges'] > split).astype(bool)


vals_tenure = dataset_raw['tenure']
mean = sum(vals_tenure)/len(vals_tenure)

dataset_raw['tenure'] = (dataset_raw['tenure'] > split).astype(bool)

vals_monthly_charges = list(dataset_raw.MonthlyCharges)

mean_1 = sum(vals_monthly_charges) /len(vals_monthly_charges)
split = binarize_using_infogain('MonthlyCharges',dataset_raw,label)
dataset_raw['MonthlyCharges'] = (dataset_raw['MonthlyCharges'] > split).astype(bool)

vals_Total_charges = list(dataset_raw.TotalCharges)

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
split = binarize_using_infogain('TotalCharges',dataset_raw,label)

dataset_raw['TotalCharges'] = (dataset_raw['TotalCharges'] > split).astype(bool)


dataset_raw.to_csv("test_Telco.csv")