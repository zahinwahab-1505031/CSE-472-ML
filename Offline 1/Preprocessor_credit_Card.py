import sklearn
import pandas as pd
import math
dataset_raw  = pd.read_csv("Datasets\creditcard\creditcard.csv")
def binarize_using_gini(Attribute_to_binarize,dataset_raw,Label):
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
    min_gini = 99999
    split_points = list(set(split_points))
    total_samples = dataset_raw.shape[0] 
    #split_points = np.array(split_points) 
    #split_points = np.unique(split_points)
    
    split_points.sort()
    print(split_points)
    final_split = split_points[0]
    for split in split_points:
        group_1 = dataset_raw[dataset_raw[Attribute_to_binarize] <split]
        group_1_class_1 = group_1[group_1[Label]=='Yes']
        group_1_class_0 = group_1[group_1[Label]=='No']
      #  print("CHECKPOINT 1")

        #print(group_1.shape[0],group_1_class_1.shape[0],group_1_class_0.shape[0])

        group_2 = dataset_raw[dataset_raw[Attribute_to_binarize] >=split]
        group_2_class_1 = group_2[group_2[Label]=='Yes']
        group_2_class_0 = group_2[group_2[Label]=='No']
       # print("CHECKPOINT 2")
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
        #print("CHECKPOINT 3")
        gini_index_group_1 = (1.0 - (group_1_class_0_prop * group_1_class_0_prop)-(group_1_class_1_prop*group_1_class_1_prop)) * (group_1.shape[0]/total_samples)
            

        gini_index_group_2 = (1.0 - (group_2_class_0_prop * group_2_class_0_prop)-(group_2_class_1_prop*group_2_class_1_prop)) * (group_2.shape[0]/total_samples)
        gini_index = gini_index_group_1+gini_index_group_2
        
        if gini_index < min_gini:
            min_gini = gini_index
            final_split = split  
         #   print("CHECKPOINT 4 = DONE")
            #print("DO")
    print("FINAL SPLIT: ",final_split)
    return final_split
#print(dataset_raw.dtypes)
#print(dataset_raw.isna().sum())
#print(dataset_raw.isnull().sum())
Attributes = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
Label ='Class'
print(dataset_raw[Label].unique())
dataset_raw[Label]= dataset_raw[Label].replace(1, 'Yes')
dataset_raw[Label]= dataset_raw[Label].replace(0, 'No')
print(dataset_raw[Label].unique())
for attr in Attributes:
    if attr != 'Time':
        print(attr)
        split = binarize_using_gini(attr,dataset_raw,Label)
        dataset_raw[attr] = (dataset_raw[attr] > split).astype(bool)
dataset_raw.to_csv("credit_card_processed.csv")