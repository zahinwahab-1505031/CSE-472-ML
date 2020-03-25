import sklearn
import pandas as pd 
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import os.path
import csv


data =[]
label = ''
df = []
initial_attributes = []
def Initialize_Global_Variables(Dataset_name):
    global data
    global label
    global df
    global initial_attributes
    if Dataset_name=='telco':
        data = pd.read_csv("test_Telco.csv")
        label = 'Churn'
        df = pd.DataFrame (data, columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
 'StreamingMovies', 'Contract', 'PaperlessBilling' ,'PaymentMethod','MonthlyCharges', 'TotalCharges','Churn'])
        initial_attributes = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
 'StreamingMovies', 'Contract', 'PaperlessBilling' ,'PaymentMethod','MonthlyCharges', 'TotalCharges']
    if Dataset_name=='CreditCard':
        data = pd.read_csv('credit_card_processed_infogain.csv')
        #data = pd.read_csv('Sampled_credit_card.csv')
        df = pd.DataFrame(data,columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class'])

        initial_attributes = [ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

        label ='Class'
    if Dataset_name == 'Adult':
        data = pd.read_csv('Adult_train.csv')
        label = 'decision'
        df = pd.DataFrame(data,columns =  ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','decision'])
        initial_attributes =  ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']


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


def binarize_using_infogain_int_split_point(Attribute_to_binarize,dataset_raw,label):
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
    #    print(split_points)
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
    #    print(split_points)
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

def calculate_entropy(attribute_name,column_value,examples):
    total_p = 0
    total_n = 0
    for iter in examples[examples[attribute_name]==column_value][label].values:
        if iter == 'Yes':
            total_p = total_p+1
        elif iter == 'No':
            total_n = total_n+1
    total = total_p+total_n
    q = total_p/total
    
    B = calculate_entropy_step2(q)
    return total_p,total_n,B


    

def calculate_information_gain(attribute_name,examples):
    #print(attribute_name)
    unique_values = examples[attribute_name].unique()
    #print(unique_values)
    sum=0
    total_samples=len(examples)
    total_samples_positive= len(examples[examples[label]=='Yes'])
    ParentEntropy = calculate_entropy_step2(total_samples_positive/total_samples)
    #print(ParentEntropy)
    for val in unique_values:

        pk,nk,B = calculate_entropy(attribute_name,val,examples)
        sum = sum + ((pk+nk)*B)/total_samples
    Entropy = ParentEntropy - sum
    #print(Entropy)
    return Entropy
def PluralityVal(examples):
    L = len(examples)
    #print(L)
    positive_samples = examples[examples[label]=='Yes']
    negative_samples = examples[examples[label]=='No']
    if len(positive_samples) >= len(negative_samples):
        return 'Yes'
    else:
        return 'No'
def check_if_same_classification(examples):
    L = len(examples)
    #print(L)
    positive_samples = examples[examples[label]=='Yes']
    negative_samples = examples[examples[label]=='No']
    if len(positive_samples) == L:
        return 'Yes'
    elif len(negative_samples) == L:
        return 'No'
    else: 
        return 'False'
def DecisionTree(examples,attributes,parent_examples,depth):
   # print(len(examples))
   # print(PluralityVal(examples))
    tree = {}
    if examples.shape[0] == 0: 
        tree['leaf'] = PluralityVal(parent_examples)
        return tree
    elif check_if_same_classification(examples) != 'False':
     #   print("case 2")
        tree['leaf'] = check_if_same_classification(examples)
        return tree
    elif len(attributes)==0 or depth==0:
     #   print("case 3")
        tree['leaf'] = PluralityVal(examples)
        return tree
    else:
        max = -math.inf
        max_attr = ''
        for attr in attributes:

            Entropy = calculate_information_gain(attr,examples)
            if Entropy > max:
                max = Entropy
                max_attr = attr
            
        #print("Attribute to be taken"+max_attr)
        
       
        
        tree['Internal'] = max_attr #list(examples[max_attr].unique())
        #print(tree)
        for vk in df[max_attr].unique():
            #print(vk)
            exs = examples.loc[examples[max_attr] == vk]
            exs.reset_index(drop=True)
            #print(exs)
            #print(attributes)
            
            if max_attr in attributes:
                attributes.remove(max_attr)

            subtree = DecisionTree(exs.copy(),attributes[:],examples.copy(),depth-1)
            tree[vk] = subtree
    
    return tree
def predict_label(Decision_Tree,examples):
    tree = Decision_Tree
    #print(tree)
    predicted_label = []
    #print("================")
    #print(examples.shape)
    #print("================")
    for i in range(examples.shape[0]):
        
        tree = Decision_Tree
        label = 'Undetermined'
        while label!='Yes' and label!='No':
            if 'Internal' in tree:

                attribute_to_check = tree['Internal']
                feature = examples[attribute_to_check][i]
                tree = tree[feature]
            if 'leaf' in tree:
                label = tree['leaf']
                #print(label)
                predicted_label.append(label)
                

 #   print(predicted_label)
    return predicted_label
def calculate_performace_summary(test_y,pred_y):
    #The accuracy can be defined as the percentage of correctly classified instances 
    # (TP + TN)/(TP + TN + FP + FN). where TP, FN, FP and TN represent the number of true positives, 
    #false negatives, false positives and true negatives, respectively
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    print(len(test_y))
    for i in range(test_y.size):
        #print(test_y[i],pred_y[i])
        if test_y[i]=='Yes':

            if pred_y[i]=='Yes':
                #print("Case : YY")
                true_positive = true_positive+1
            elif pred_y[i]=='No':
                #print("Case : YN")
                false_negative = false_negative+1
        elif test_y[i]=='No':
            
            if pred_y[i]=='No':
                #print("Case : NN")
                true_negative = true_negative+1
            elif pred_y[i]=='Yes': 
                #print("Case : NY")
                false_positive = false_positive+1

    Accuracy = ((true_positive + true_negative)*1.0)/(true_positive + true_negative + false_positive + false_negative)
    Accuracy = Accuracy*100
    Recall = (true_positive*1.0) / (true_positive + false_negative) #true positive rate
    
    Specificity = (true_negative*1.0)/ (true_negative+false_positive) #true negative rate
    Precision = (true_positive*1.0) / (true_positive + false_positive)
    false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
    f1score = 2.0/((1.0/Recall)+(1.0/Precision))
    print("Total: ",true_positive + true_negative + false_positive + false_negative)
    
    print("Accuracy: ", Accuracy,"%")
    print("True Positive: ",true_positive )
    print("True Negative: ",true_negative)
    print("False Positive: ",false_positive )
    print("False Negative: ",false_negative)
    print("Recall: ", Recall)
    print("Specificity: ", Specificity)
    print("Precision: ", Precision)
    print("False Discovery Rate: ", false_discovery_rate)
    print("F1 Score: ", f1score)
def calculate_performace(test_y,pred_y):
    #The accuracy can be defined as the percentage of correctly classified instances 
    # (TP + TN)/(TP + TN + FP + FN). where TP, FN, FP and TN represent the number of true positives, 
    #false negatives, false positives and true negatives, respectively
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    print(len(test_y))
    for i in range(test_y.size):
        #print(test_y[i],pred_y[i])
        if test_y[i]=='Yes':

            if pred_y[i]=='Yes':
                #print("Case : YY")
                true_positive = true_positive+1
            elif pred_y[i]=='No':
                #print("Case : YN")
                false_negative = false_negative+1
        elif test_y[i]=='No':
            
            if pred_y[i]=='No':
                #print("Case : NN")
                true_negative = true_negative+1
            elif pred_y[i]=='Yes': 
                #print("Case : NY")
                false_positive = false_positive+1

    Accuracy = ((true_positive + true_negative)*1.0)/(true_positive + true_negative + false_positive + false_negative)
    Accuracy = Accuracy*100
    #Recall = (true_positive*1.0) / (true_positive + false_negative) #true positive rate
    
    #Specificity = (true_negative*1.0)/ (true_negative+false_positive) #true negative rate
    #Precision = (true_positive*1.0) / (true_positive + false_positive)
    #false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
    #f1score = 2.0/((1.0/Recall)+(1.0/Precision))
    print("Total: ",true_positive + true_negative + false_positive + false_negative)
    
    print("Accuracy: ", Accuracy,"%")
    print("True Positive: ",true_positive )
    print("True Negative: ",true_negative)
    print("False Positive: ",false_positive )
    print("False Negative: ",false_negative)
    #print("Recall: ", Recall)
    #print("Specificity: ", Specificity)
    #print("Precision: ", Precision)
    #print("False Discovery Rate: ", false_discovery_rate)
    #print("F1 Score: ", f1score)

def Adaboost(examples,K):
    w = []
    h = []
    Z = []
    for i in range(examples.shape[0]):
        w.append(1.0/examples.shape[0])
    #print(w)
    for k in range(K):
        #print(k,"th BOOSTER")
        data_to_be_sampled = examples.copy()
        #np.random.seed(seed=10)
        data = data_to_be_sampled.sample(frac=1,weights=w,replace=True,random_state=1)
        data.reset_index(drop=True)
        data_to_pass = data.copy()
        parent_data_to_pass = data.copy()
        attributes = initial_attributes.copy()
        tree = DecisionTree(data_to_pass,attributes,parent_data_to_pass,1)
        
        pred_y = predict_label(tree,examples)
        true_y = examples[label]
        error = 0
        for j in range(examples.shape[0]):
            if pred_y[j]!=true_y[j]:
                error = error+w[j]
        #print("ERROR: ",error)
        if error > 0.5:
            continue
       # print(tree)
        #print("H: " ,tree)
        if error == 0.0:
            error = 0.0000000000000000000001
        for j in range(examples.shape[0]):
            if pred_y[j]==true_y[j]:
                w[j] = w[j]*(error/(1-error))
        #print("before:")
        #print(sum(w))
        summation_w = sum(w)
        for iter in range(len(w)):
            w[iter] = w[iter]/summation_w
        h.append(tree)
        var = (1-error)/error
        Z.append(math.log(var,2.0))
        #print("Z: ",math.log(var,2.0))
        #print("after normalizing")
        #print(sum(w))
    return h,Z

def encode(rawdata):
    encoded_data = []
    for iter in range(len(rawdata)):
        if rawdata[iter]=='Yes':
            encoded_data.append(1.0)
        elif rawdata[iter]=='No':
            encoded_data.append(-1.0)
    
    return encoded_data

def predict_label_adaboost(h,Z,examples):
    
    final_encoded_results = []
    for iter in range(len(h)):
        #print(Z[iter])
        pred_y = predict_label(h[iter],examples)
        pred_y_encoded = encode(pred_y)
        final_encoded_results.append(pred_y_encoded)
    weighted_result = []
    for i in range(examples.shape[0]):
        res = 0
        for j in range(len(Z)):
            res = res + final_encoded_results[j][i]*Z[j]
        

        weighted_result.append(res)
    
    pred_label = []
    for i in range(len(weighted_result)):
        
        if weighted_result[i]>= 0:
            pred_label.append('Yes')
        else:
            pred_label.append('No')
    return pred_label

def Preprocess_Adult_train():
    file1=open('Datasets/Adult/adult.data.txt')
    In_text = csv.reader(file1,delimiter = ',')
    
    file2 =open('adult.csv','w')
    out_csv = csv.writer(file2)
    
    file3 = out_csv.writerows(In_text)
    
    file1.close()
    file2.close()

    df = pd.read_csv('adult.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
    'native-country','decision'])
    #print(df.head())
    #print(df.shape)
    #print(len(df['capital-gain'].unique()))

    numerical_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']
    df['decision']= df['decision'].replace(' >50K', 'Yes')
    df['decision']= df['decision'].replace(' <=50K', 'No')
    label = 'decision'
    SplitPoints = []
    for attr in numerical_features:
        print(attr)
        split = binarize_using_infogain(attr,df,label)
        SplitPoints.append(split)
        df[attr] = (df[attr] > split).astype(bool)
    df['workclass']= df['workclass'].replace(' ?', df['workclass'].value_counts().idxmax())
    df['occupation']= df['occupation'].replace(' ?', df['occupation'].value_counts().idxmax())
    df['native-country']= df['native-country'].replace(' ?', df['native-country'].value_counts().idxmax())
    df.to_csv('Adult_train.csv')
    return SplitPoints

def Preprocess_Adult_test(Splitpoints):
    file1=open('Datasets/Adult/adult.test.txt')
    In_text = csv.reader(file1,delimiter = ',')
    
    file2 =open('adult_test_unfiltered.csv','w')
    out_csv = csv.writer(file2)
    
    file3 = out_csv.writerows(In_text)
    
    file1.close()
    file2.close()

    df = pd.read_csv('adult_test_unfiltered.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','decision'])

    numerical_features = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week','education-num']
    print(df['decision'].value_counts())
    df['decision']= df['decision'].replace(' >50K.', 'Yes')
    df['decision']= df['decision'].replace(' <=50K.', 'No')
    label = 'decision'
    print(df['decision'].value_counts())
    i = 0
    for attr in numerical_features:
        
        df[attr] = (df[attr] > Splitpoints[i]).astype(bool)
        i = i+1

    
    df['workclass']= df['workclass'].replace(' ?', df['workclass'].value_counts().idxmax())

    #print(df['occupation'].value_counts())
    df['occupation']= df['occupation'].replace(' ?', df['occupation'].value_counts().idxmax())
    #print(df['occupation'].value_counts())

    df['native-country']= df['native-country'].replace(' ?', df['native-country'].value_counts().idxmax())

    df.to_csv('Adult_test.csv')

def Preprocess_Telco():
    dataset_raw = pd.read_csv("Datasets\\telco-customer-churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    Attributes = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling' ,'PaymentMethod','MonthlyCharges', 'TotalCharges']
    label ='Churn'


    split = binarize_using_infogain('tenure',dataset_raw,label)


    dataset_raw['tenure'] = (dataset_raw['tenure'] > split).astype(bool)

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
            
    dataset_raw['TotalCharges'] = vals_Total_charges
    split = binarize_using_infogain('TotalCharges',dataset_raw,label)

    dataset_raw['TotalCharges'] = (dataset_raw['TotalCharges'] > split).astype(bool)


    dataset_raw.to_csv("test_Telco.csv")

def Preprocess_Credit_Card():
    dataset_raw  = pd.read_csv("Datasets\creditcard\creditcard.csv")
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
            split = binarize_using_infogain_int_split_point(attr,dataset_raw,label)
            dataset_raw[attr] = (dataset_raw[attr] > split).astype(bool)
    dataset_raw.to_csv("credit_card_processed_infogain.csv")
def Sample_Credit_Card():
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
    
    df_no = df[df[label]=='No'].sample(n=20000,random_state=1)
    df_no.reset_index(drop=True)
    print(df_no.head())
    print(df_no.shape)

    df_new = pd.concat([df_yes, df_no], axis=0)

    df_new = df_new.sample(frac=1,random_state=1).reset_index(drop=True)

    for attr in Attributes:
        if attr != 'Time':
            print(attr)
            split = binarize_using_infogain(attr,df_new,label)
            df_new[attr] = (df_new[attr] > split).astype(bool)



    print(df_new.head())
    print(df_new.shape)

    df_new.to_csv('Sampled_credit_card.csv')
def runTelco_CreditCard():
    #print(data)
    y = data[label] # define the target variable (dependent variable) as y
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2,shuffle=False)
    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)

    #
    #print("Decision Tree")
    #print(Decision_Tree)
    for rounds in range(5,25,5):
        
        print("ROUND: ",rounds)
        X_train =X_train.reset_index(drop=True)
        h,Z = Adaboost(X_train,rounds)
        print("=============TRAIN ACCURACY=======================")
        y_train = X_train[label]
        y_pred = predict_label_adaboost(h,Z,X_train)
        calculate_performace(y_train,y_pred)
        print("=============TEST ACCURACY=======================")
        X_test =X_test.reset_index(drop=True)
        
        y_test =X_test[label]
        y_pred = predict_label_adaboost(h,Z,X_test)
        calculate_performace(y_test,y_pred)

    #this works fine
    print("==========DECISION TREE VERSION============================")
    Decision_Tree= DecisionTree(X_train,initial_attributes,X_train,35)
    X_test =pd.DataFrame.reset_index(X_test)
    #X_test = X_test.drop(labels = ['Index'],axis=1)
    #print(X_test)
    print("=================TEST ACCURACY=======================")
    pred_y = predict_label(Decision_Tree,X_test)
    y_test = X_test[label]
    #y_test = pd.DataFrame.reset_index(y_test)
    calculate_performace_summary(y_test,pred_y)


    print("=================TRAIN ACCURACY=======================")
    pred_y = predict_label(Decision_Tree,X_train)
    y_train = X_train[label]
    #y_test = pd.DataFrame.reset_index(y_test)
    calculate_performace_summary(y_train,pred_y)
def runAdult_test_train_full():
    
    
    #print("Decision Tree")
    #print(Decision_Tree)
    data_test = pd.read_csv('Adult_test.csv')
    df_test =pd.read_csv('Adult_test.csv')
    # pd.DataFrame(data_test,columns =  ['age','workclass','fnlwgt','education','education-num','marital-status',
    #'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','decision'])

    '''print("-----------------DECISION TREE VERSION-----------------------")
    Decision_Tree= DecisionTree(df,initial_attributes,df,40)
    print("=====================TEST ACCURACYYYYY===================")
    pred_y = predict_label(Decision_Tree,df_test)
    y_test = df_test[label]
    calculate_performace_summary(y_test,pred_y)
    print("=====================TRAIN ACCURACYYYYY===================")
    pred_y = predict_label(Decision_Tree,df)
    y_train = df[label]
    calculate_performace_summary(y_train,pred_y)'''
    print("-----------------ADABOOST VERSION-----------------------")
    for rounds in range(5,25,5):
        print("ROUND: ",rounds)
        df_test = pd.read_csv('Adult_test.csv')
        h,Z = Adaboost(df,rounds)
        print("=============TRAIN ACCURACY=======================")
        y_train = df[label]
        y_pred = predict_label_adaboost(h,Z,df)
        calculate_performace(y_train,y_pred)
        print("=============TEST ACCURACY=======================")

        y_test =df_test[label]
        y_pred = predict_label_adaboost(h,Z,df_test)
        calculate_performace(y_test,y_pred)
    print("-----------------DECISION TREE VERSION-----------------------")
    Decision_Tree= DecisionTree(df,initial_attributes,df,40)
    print("=====================TEST ACCURACYYYYY===================")
    pred_y = predict_label(Decision_Tree,df_test)
    y_test = df_test[label]
    calculate_performace_summary(y_test,pred_y)
    print("=====================TRAIN ACCURACYYYYY===================")
    pred_y = predict_label(Decision_Tree,df)
    y_train = df[label]
    calculate_performace_summary(y_train,pred_y)
def runTelco():
    Preprocess_Telco()
    Initialize_Global_Variables('telco')
    runTelco_CreditCard()
def runCredit_Card():
   
    Sample_Credit_Card()
    #Preprocess_Credit_Card()
    Initialize_Global_Variables('CreditCard')
    runTelco_CreditCard()
def runAdult():
    splits = Preprocess_Adult_train()
    Preprocess_Adult_test(splits)
    Initialize_Global_Variables('Adult')
    runAdult_test_train_full()


runTelco()