import sklearn
import pandas as pd
import math
dataset_raw  = pd.read_csv("Datasets\creditcard\creditcard.csv")
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
    res = math.floor(values_to_binarize[0]-abs((values_to_binarize[0]-values_to_binarize[1])/2))
    split_points.append(res)
    for iter in range(len(values_to_binarize)-1):
        res = math.floor((values_to_binarize[iter]+values_to_binarize[iter+1])/2)
        split_points.append(res)
    res = math.floor(values_to_binarize[len(values_to_binarize)-1]+((values_to_binarize[len(values_to_binarize)-1]-values_to_binarize[len(values_to_binarize)-2])/2))
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
#print(dataset_raw.dtypes)
#print(dataset_raw.isna().sum())
#print(dataset_raw.isnull().sum())
Attributes = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
label ='Class'
print(dataset_raw[label].unique())
dataset_raw[label]= dataset_raw[label].replace(1, 'Yes')
dataset_raw[label]= dataset_raw[label].replace(0, 'No')
print(dataset_raw[label].unique())
for attr in Attributes:
    if attr != 'Time':
        print(attr)
        split = binarize_using_infogain(attr,dataset_raw,label)
        dataset_raw[attr] = (dataset_raw[attr] > split).astype(bool)
dataset_raw.to_csv("credit_card_processed_infogain.csv")