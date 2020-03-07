import pandas as pd 
import sklearn
import math
    
data = pd.read_csv("Tennis.csv")
#print(data.head())
df = pd.DataFrame (data, columns = ['Day','Outlook','Temperature','Humidity','Wind','Play'])
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
    for iter in examples[examples[attribute_name]==column_value]['Play'].values:
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
    total_samples_positive= len(examples[examples['Play']=='Yes'])
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
    positive_samples = examples[examples['Play']=='Yes']
    negative_samples = examples[examples['Play']=='No']
    if len(positive_samples) > len(negative_samples):
        return 'Yes'
    else:
        return 'No'
def check_if_same_classification(examples):
    L = len(examples)
    #print(L)
    positive_samples = examples[examples['Play']=='Yes']
    negative_samples = examples[examples['Play']=='No']
    if len(positive_samples) == L:
        return 'Yes'
    elif len(negative_samples) == L:
        return 'No'
    else: 
        return 'False'
def DecisionTree(examples,attributes,parent_examples):
    #print(len(examples))
   # print(PluralityVal(examples))
    tree = {}
    if examples.empty: 
        tree['leaf'] = PluralityVal(parent_examples)
        return tree
    elif check_if_same_classification(examples) != 'False':
        tree['leaf'] = check_if_same_classification(examples)
        return tree
    elif attributes is None:
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
            #print(exs)
            #print(attributes)
            
            if max_attr in attributes:
                attributes.remove(max_attr)

            subtree = DecisionTree(exs,attributes,examples)
            tree[vk] = subtree
    
    return tree
def predict_label(Decision_Tree,examples):
    tree = Decision_Tree
    print(tree)
    predicted_label = []
    print(df.shape)
    for i in range(df.shape[0]):
        tree = Decision_Tree
        label = 'Undetermined'
        while label!='Yes' and label!='No':
            attribute_to_check = tree['Internal']
            feature = examples[attribute_to_check][i]
            tree = tree[feature]
            if 'leaf' in tree:
                label = tree['leaf']
                print(label)
                predicted_label.append(label)
                break

    print(predicted_label)
    return predicted_label
def calculate_performace(test_y,pred_y):
    #The accuracy can be defined as the percentage of correctly classified instances 
    # (TP + TN)/(TP + TN + FP + FN). where TP, FN, FP and TN represent the number of true positives, 
    #false negatives, false positives and true negatives, respectively
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(test_y.size):
        if test_y[i]=='Yes':

            if pred_y[i]=='Yes':
                true_positive = true_positive+1
            elif pred_y[i]=='No':
                false_negative = false_negative+1
        elif test_y[i]=='No':
            if pred_y[i]=='No':
                true_negative = true_negative+1
            elif pred_y[i]=='Yes': 
                false_positive = false_positive+1

    Accuracy = ((true_positive + true_negative)*1.0)/(true_positive + true_negative + false_positive + false_negative)
    Recall = (true_positive*1.0) / (true_positive + false_negative) #true positive rate
    
    Specificity = (true_negative*1.0)/ (true_negative+false_positive) #true negative rate
    Precision = (true_positive*1.0) / (true_positive + false_positive)
    false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
    f1score = 2.0/((1.0/Recall)+(1.0/Precision))
    print(Accuracy)
    print(Recall)
    print(Specificity)
    print(Precision)
    print(false_discovery_rate)
    print(f1score)

                

initial_attributes = ['Outlook','Temperature','Humidity','Wind']
Decision_Tree= DecisionTree(df,initial_attributes,df)
print("Decision Tree")
print(Decision_Tree)
pred_y = predict_label(Decision_Tree,df)
test_y = df['Play']
calculate_performace(test_y,pred_y)





