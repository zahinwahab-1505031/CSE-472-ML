#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random


# In[2]:


with open('data.txt') as file:
    data = np.array([np.array([float(digit) for digit in line.split()]) for line in file])

print(data.shape)


# In[3]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(data)
#print(X_train)
#print(X_std)
print(X_std.shape)
mean_vec = np.mean(X_std, axis=0)
cov_mat =( (X_std - mean_vec).T.dot((X_std - mean_vec)) )/ (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[4]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
k = 0
eig_vec_top_2 = np.zeros((2,data.shape[1]))
for i in eig_pairs[:2]:
    print(i[0])
    eig_vec_top_2 [k] = i[1]
    k = k+1


# In[5]:


projection_matrix = (eig_vec_top_2.T[:][:]).T


# In[6]:


X_pca = X_std.dot(projection_matrix.T)
X_pca


# In[7]:


plt.scatter(X_pca.T[0],X_pca.T[1])

plt.savefig('pcaplot.png')
plt.show()


# In[8]:



K = 3
D = 2
np.random.seed(42)

means = np.zeros((K,D))
covariances = np.zeros((K,D,D))
#covariances =np.random.randn(K,D,D)
weights =np.array([0.33,0.33,0.34])

from numpy import random


for k in range(K):
    x = random.randint(1000)
    means[k] = X_pca[x]

for k in range(K):
    x = random.randint(1000)
    
    covariances[k]=( (X_pca - means[k]).T.dot((X_pca - means[k])) )/ (X_pca.shape[0]-1)
print("Initialization")
print("Means:\n",means)
print("Co-variances:\n",covariances)
print("weights:\n",weights)
Nk_xi_arr = np.zeros((data.shape[0],K))

for i in range(data.shape[0]):
    for k in range(K):
        
        temp2 = math.sqrt(pow(2*3.1416,D)*abs(np.linalg.det(covariances[k]) ))
        x_i = X_pca[i]
        temp3 = np.linalg.inv(covariances[k])
        row = (x_i - means[k]).reshape(-1,1)
        temp = np.matmul(row.T,(np.matmul(temp3,row)))
        Nk_xi_arr[i][k]=(1/temp2)*np.exp(-0.5*temp)



loglikelihood = 0

for i in range(data.shape[0]):
    sum = 0
    for k in range(K):
        sum = sum + weights[k]*Nk_xi_arr[i,k]
    loglikelihood = loglikelihood + math.log(sum)
print(loglikelihood)
loglikelihood = abs(loglikelihood)
def EM_Algorithm(means,covariances,weights,loglikelihood,Nk_xi_arr,K,D):
   # print("In EM algo function:=============================")
    #print("Means:\n",means)
    #print("Co-variances:\n",covariances)
    #print("weights:\n",weights)
    print("Before starting EM algo:",loglikelihood)
    iteration = 1
    while(True):
        print("==================Iteration:",iteration,"=========================")
        iteration = iteration + 1
        
        P_ik_arr = np.zeros((data.shape[0],K))
       # print(P_ik_arr)
        for i in range(data.shape[0]):
           # denom = np.sum(np.multiply(weights.reshape(1,-1),Nk_xi_arr[i].reshape(1,-1)))
            denominator = 0
            for k in range(K):
                denominator =  denominator + (Nk_xi_arr[i][k]*weights[k])
            for k in range(K):
                P_ik_arr[i][k] = (Nk_xi_arr[i][k]*weights[k]) / denominator
        
       # print(P_ik_arr)
       # print("checking mean")
        for k in range(K):
          #  print(P_ik_arr[:,k].reshape(-1,1)[0])
          #  print(X_pca[0])
          #  print(np.multiply(P_ik_arr[:,k].reshape(-1,1),X_pca)[0])
            numerator = np.sum(np.multiply(P_ik_arr[:,k].reshape(-1,1),X_pca),axis=0)
            denominator = np.sum(P_ik_arr[:,k].reshape(-1,1))
            means[k] = numerator/denominator
        for k in range(K):
            numerator = np.zeros((D,D))
            for i in range(data.shape[0]):
                row = X_pca[i]  - means[k]
                temp = (row.reshape(-1,1))
                temp2 = (row.reshape(1,-1))
                #print("covar calc:",temp,"*",temp2,"=",np.matmul(temp,temp2))
                numerator = numerator + P_ik_arr[i,k]*(np.matmul(temp,temp2))

            denominator = np.sum(P_ik_arr[:,k].reshape(-1,1))
            covariances[k] = numerator/denominator
        N = data.shape[0]
        for k in range(K):
            numerator = np.sum(P_ik_arr[:,k].reshape(-1,1))
            weights[k] = numerator/N
      #  print("Means:\n",means)
       # print("Co-variances:\n",covariances)
        #print("weights:\n",np.sum(weights))
        Nk_xi_arr = np.zeros((data.shape[0],K))
        for i in range(data.shape[0]):
            for k in range(K):
                temp2 = math.sqrt(pow(2*3.1416,D)*abs(np.linalg.det(covariances[k]) ))

                x_i = X_pca[i]
                temp3 = np.linalg.inv(covariances[k])
                row = (x_i - means[k]).reshape(-1,1)
                
                temp = np.matmul(row.T,(np.matmul(temp3,row)))
                Nk_xi_arr[i][k]=(1/temp2)*np.exp(-0.5*temp)
        previous_loglikelihood = loglikelihood
        loglikelihood = 0

        for i in range(data.shape[0]):
            sum = 0
            for k in range(K):
                sum = sum + weights[k]*Nk_xi_arr[i,k]
            loglikelihood = loglikelihood + math.log(sum)
        
        loglikelihood = abs(loglikelihood)
        print("Loglikelihood:",loglikelihood)
        diff = loglikelihood - previous_loglikelihood
      #  print("difference: ",diff)
        #if iteration == 10:
        if abs(diff) <= pow(10,-6):
            print("Final:")
            print("Means:\n",means)
            print("Co-variances:\n",covariances)
            print("weights:\n",weights)
            break
   # print(P_ik_arr)
    return P_ik_arr
        
    
Final_Probability = EM_Algorithm(means,covariances,weights,loglikelihood,Nk_xi_arr,K,D)    
from scipy.special import softmax
softmax_arr = softmax(Final_Probability, axis=1)
boolean_arr = np.zeros_like(softmax_arr)
boolean_arr[np.arange(len(softmax_arr)), softmax_arr.argmax(1)] = 1
color = []
for i in range(data.shape[0]):
    idx = np.where(boolean_arr[i]==1)[0][0]
    if idx == 0:
        color.append( 'green')
    elif idx == 1:
        
        color.append('red')
    elif idx == 2:
        color.append('blue')
  
plt.scatter(X_pca.T[0],X_pca.T[1],c=color)
plt.savefig('em_pca.png')


# In[ ]:





# In[9]:



  # plt.show()


# In[ ]:




