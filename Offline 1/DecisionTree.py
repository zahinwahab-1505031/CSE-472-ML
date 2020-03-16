import pandas as pd 
import sklearn
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#print(data.head())
'''
data = pd.read_csv("Tennis.csv")
label = 'Play'
initial_attributes = ['Day','Outlook','Temperature','Humidity','Wind']
df = pd.DataFrame (data, columns = ['Day','Outlook','Temperature','Humidity','Wind','Play'])

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
'''
data = pd.read_csv('Adult_train.csv')
label = 'decision'
df = pd.DataFrame(data,columns =  ['age','workclass','fnlwgt','education','education-num','marital-status',
'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','decision'])
initial_attributes =  ['age','workclass','fnlwgt','education','education-num','marital-status',
'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
'''
data = pd.read_csv('credit_card_processed_infogain.csv')
df = pd.DataFrame(data,columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class'])

initial_attributes = [ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
label ='Class'
#df[label]= df[label].replace(1, 'Yes')
#df[label]= df[label].replace(0, 'No')
'''
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
    Recall = (true_positive*1.0) / (true_positive + false_negative) #true positive rate
    
    Specificity = (true_negative*1.0)/ (true_negative+false_positive) #true negative rate
    Precision = (true_positive*1.0) / (true_positive + false_positive)
    false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
    f1score = 2.0/((1.0/Recall)+(1.0/Precision))
    print("Total: ",true_positive + true_negative + false_positive + false_negative)
    
    print("Accuracy: ", Accuracy)
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
    #Recall = (true_positive*1.0) / (true_positive + false_negative) #true positive rate
    
    #Specificity = (true_negative*1.0)/ (true_negative+false_positive) #true negative rate
    #Precision = (true_positive*1.0) / (true_positive + false_positive)
    #false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
    #f1score = 2.0/((1.0/Recall)+(1.0/Precision))
    print("Total: ",true_positive + true_negative + false_positive + false_negative)
    
    print("Accuracy: ", Accuracy)
    print("True Positive: ",true_positive )
    print("True Negative: ",true_negative)
    print("False Positive: ",false_positive )
    print("False Negative: ",false_negative)
    #print("Recall: ", Recall)
    #print("Specificity: ", Specificity)
    #print("Precision: ", Precision)
    #print("False Discovery Rate: ", false_discovery_rate)
    #print("F1 Score: ", f1score)
def dfs(tree):
    if type(tree) == str:
        print(tree)
        return
    
    for keys in tree.keys():
        print(keys)
        dfs(tree[keys])
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
        data = data_to_be_sampled.sample(frac=1,weights=w,replace=True)
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
    print(len(h),len(Z))
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




'''
#for telco

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
'''
#for adult


Decision_Tree= DecisionTree(df,initial_attributes,df,40)
#print("Decision Tree")
#print(Decision_Tree)
data_test = pd.read_csv('Adult_test.csv')
df_test =pd.read_csv('Adult_test.csv')
# pd.DataFrame(data_test,columns =  ['age','workclass','fnlwgt','education','education-num','marital-status',
#'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','decision'])

print("-----------------DECISION TREE VERSION-----------------------")
print("=====================TEST ACCURACYYYYY===================")
pred_y = predict_label(Decision_Tree,df_test)
y_test = df_test[label]
calculate_performace_summary(y_test,pred_y)
print("=====================TRAIN ACCURACYYYYY===================")
pred_y = predict_label(Decision_Tree,df)
y_train = df[label]
calculate_performace_summary(y_train,pred_y)
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
