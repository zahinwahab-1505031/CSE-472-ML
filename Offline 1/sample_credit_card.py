import pandas as pd
import sklearn

df = pd.read_csv('credit_card_processed.csv')
label = 'Class'
df_yes = df[df[label]=='Yes']
df_yes.reset_index(drop=True)
print(df_yes.head())
print(df_yes.shape)
df_no = df[df[label]=='No'].sample(n=20000)
df_no.reset_index(drop=True)
print(df_no.head())
print(df_no.shape)

df_new = pd.concat([df_yes, df_no], axis=0)

df_new = df_new.sample(frac=1).reset_index(drop=True)

print(df_new.head())
print(df_new.shape)

df_new.to_csv('Sampled_credit_card.csv')