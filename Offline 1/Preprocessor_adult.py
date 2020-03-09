import sklearn
import pandas as pd
import os.path
import csv
 
'''file1=open('Datasets/Adult/adult.data.txt')
In_text = csv.reader(file1,delimiter = ',')
 
file2 =open('adult.csv','w')
out_csv = csv.writer(file2)
 
file3 = out_csv.writerows(In_text)
 
file1.close()
file2.close()'''

df = pd.read_csv('adult.csv', names = ['age','workclass','fnlwgt','education','education-num','marital-status',
'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','decision'])
#print(df.head())
print(df.shape)
print(len(df['capital-gain'].unique()))


vals_age = list(df.age)
#print(unique_vals_tenure)
mean = sum(vals_age)/len(vals_age)
print(mean)
df['age'] = (df['age'] > mean).astype(bool)


vals_fnlwgt = list(df.fnlwgt)
#print(unique_vals_tenure)
mean = sum(vals_fnlwgt)/len(vals_fnlwgt)
print(mean)
df['fnlwgt'] = (df['fnlwgt'] > mean).astype(bool)

'''capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.'''
vals_capital_gain = list(df['capital-gain'])
#print(unique_vals_tenure)
mean = sum(vals_capital_gain)/len(vals_capital_gain)
print(mean)
df['capital-gain'] = (df['capital-gain'] > mean).astype(bool)

vals_capital_loss = list(df['capital-loss'])
mean = sum(vals_capital_loss)/len(vals_capital_loss)
print(mean)
df['capital-loss'] = (df['capital-loss'] > mean).astype(bool)

vals_hours_per_week = list(df['hours-per-week'])
mean = sum(vals_hours_per_week)/len(vals_hours_per_week)
print(mean)
df['hours-per-week'] = (df['hours-per-week'] > mean).astype(bool)

vals_education_num = list(df['education-num'])
mean = sum(vals_education_num)/len(vals_education_num)
print(mean)
df['education-num'] = (df['education-num'] > mean).astype(bool)

#print(df['workclass'].value_counts())
df['workclass']= df['workclass'].replace(' ?', df['workclass'].value_counts().idxmax())
#print(df['workclass'].value_counts())


#print(df['education'].value_counts())
#print(df['marital-status'].value_counts())
#print(df['occupation'].value_counts())
df['occupation']= df['occupation'].replace(' ?', df['occupation'].value_counts().idxmax())
#print(df['occupation'].value_counts())
#print(df['relationship'].value_counts())
#print(df['race'].value_counts())
#print(df['sex'].value_counts())
df['native-country']= df['native-country'].replace(' ?', df['native-country'].value_counts().idxmax())
#print(df['native-country'].value_counts())
#print(df['decision'].value_counts())
df['decision']= df['decision'].replace(' >50K', 'Yes')
df['decision']= df['decision'].replace(' <=50K', 'No')
#print(df['decision'].value_counts())
df.to_csv('Adult_train.csv')
