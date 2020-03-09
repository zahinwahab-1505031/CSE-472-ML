import pandas as pd 
import sklearn
import math
from sklearn.model_selection import train_test_split

#print(data.head())
'''
data = pd.read_csv("Tennis.csv")
label = 'Play'
initial_attributes = ['Day','Outlook','Temperature','Humidity','Wind']
df = pd.DataFrame (data, columns = ['Day','Outlook','Temperature','Humidity','Wind','Play'])
'''
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
def DecisionTree(examples,attributes,parent_examples):
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
    elif len(attributes)==0:
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

            subtree = DecisionTree(exs.copy(),attributes[:],examples.copy())
            tree[vk] = subtree
    
    return tree
def predict_label(Decision_Tree,examples):
    tree = Decision_Tree
    #print(tree)
    predicted_label = []
    print("================")
    print(examples.shape)
    print("================")
    for i in range(examples.shape[0]):
        
        tree = Decision_Tree
        label = 'Undetermined'
        while label!='Yes' and label!='No':
            attribute_to_check = tree['Internal']
            feature = examples[attribute_to_check][i]
            tree = tree[feature]
            if 'leaf' in tree:
                label = tree['leaf']
                #print(label)
                predicted_label.append(label)
                

 #   print(predicted_label)
    return predicted_label
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
    Recall = (true_positive*1.0) / (true_positive + false_negative) #true positive rate
    
    Specificity = (true_negative*1.0)/ (true_negative+false_positive) #true negative rate
    Precision = (true_positive*1.0) / (true_positive + false_positive)
    false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
    f1score = 2.0/((1.0/Recall)+(1.0/Precision))
    print("Total: ")
    print(true_positive + true_negative + false_positive + false_negative)
    print(Accuracy)
    print(Recall)
    print(Specificity)
    print(Precision)
    print(false_discovery_rate)
    print(f1score)
#df = pd.DataFrame(diabetes.data, columns=columns) # load the dataset as a pandas data frame
#for train-test split in telco
def dfs(tree):
    if type(tree) == str:
        print(tree)
        return
    
    for keys in tree.keys():
        print(keys)
        dfs(tree[keys])

y = data[label] # define the target variable (dependent variable) as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1,shuffle=False)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

Decision_Tree= DecisionTree(X_train,initial_attributes,X_train)
print("Decision Tree")
print(Decision_Tree)
#this works fine

X_test =pd.DataFrame.reset_index(X_test)
#X_test = X_test.drop(labels = ['Index'],axis=1)
#print(X_test)
print("=================TEST ACCURACY=======================")
pred_y = predict_label(Decision_Tree,X_test)
y_test = X_test[label]
#y_test = pd.DataFrame.reset_index(y_test)
calculate_performace(y_test,pred_y)


print("=================TRAIN ACCURACY=======================")
pred_y = predict_label(Decision_Tree,X_train)
y_test = X_train[label]
#y_test = pd.DataFrame.reset_index(y_test)
calculate_performace(y_test,pred_y)



#for adult
'''

Decision_Tree= DecisionTree(df,initial_attributes,df)
print("Decision Tree")
print(Decision_Tree)
df_test = pd.read_csv('Adult_test.csv')
print("=====================TEST ACCURACYYYYY===================")
pred_y = predict_label(Decision_Tree,df_test)
y_test = df_test[label]
calculate_performace(y_test,pred_y)
print("=====================TRAIN ACCURACYYYYY===================")
pred_y = predict_label(Decision_Tree,df)
y_test = df[label]
calculate_performace(y_test,pred_y)'''