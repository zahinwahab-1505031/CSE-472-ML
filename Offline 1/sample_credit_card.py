import pandas as pd
import sklearn
import math
import random
def binarize_using_gini(Attribute_to_binarize,df,label):
    values_to_binarize = list(df[Attribute_to_binarize].unique())
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
    total_samples = df.shape[0] 
    #split_points = np.array(split_points) 
    #split_points = np.unique(split_points)
    
    split_points.sort()
    print(split_points)
    final_split = split_points[0]
    for split in split_points:
        group_1 = df[df[Attribute_to_binarize] <split]
        group_1_class_1 = group_1[group_1[label]=='Yes']
        group_1_class_0 = group_1[group_1[label]=='No']

        group_2 = df[df[Attribute_to_binarize] >=split]
        group_2_class_1 = group_2[group_2[label]=='Yes']
        group_2_class_0 = group_2[group_2[label]=='No']
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
df  = pd.read_csv("Datasets\creditcard\creditcard.csv")
Attributes = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
label ='Class'

df[label]= df[label].replace(1, 'Yes')
df[label]= df[label].replace(0, 'No')
print(df['Class'].value_counts())
df_yes = df[df[label]=='Yes']
df_yes.reset_index(drop=True)
print(df_yes.head())
print(df_yes.shape)
random.seed(9001)
df_no = df[df[label]=='No'].sample(n=20000)
df_no.reset_index(drop=True)
print(df_no.head())
print(df_no.shape)

df_new = pd.concat([df_yes, df_no], axis=0)

df_new = df_new.sample(frac=1).reset_index(drop=True)

for attr in Attributes:
    if attr != 'Time':
        print(attr)
        split = binarize_using_gini(attr,df_new,label)
        df_new[attr] = (df_new[attr] > split).astype(bool)



print(df_new.head())
print(df_new.shape)

df_new.to_csv('Sampled_credit_card.csv')