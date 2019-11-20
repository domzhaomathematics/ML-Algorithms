#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("ok")
# Any results you write to the current directory are saved as output.


# In[ ]:





# **TASK A: Matrix Standardization** (We're standardizing for every columns)

# In[ ]:


def standardize(data_matrix):
    centered_data=data_matrix-np.mean(data_matrix,axis=0)
    std_data=centered_data/np.std(centered_data,axis=0)
    return std_data


# In[7]:


num_samples=5
num_features=10
np.random.seed(1)
matrix=np.random.rand(num_samples,num_features)
print(standardize(matrix))


# In[ ]:





# **TASK B: Pairwise distances in the plane**

# In[ ]:


def pairwise_distances(points1,points2):
    distances=np.empty((len(points1),len(points2)))
    for i in range(len(points1)):
        for j in range(len(points2)):
            distances[i][j]=np.linalg.norm(points1[i]-points2[j])
    return distances


# In[9]:


points1=np.array([[1,2],[1,4],[1,2]])
points2=np.array([[2,4],[2,2],[8,9],[0,0]])

np.random.seed(2)
points1=np.random.rand(3,2)
points2=np.random.rand(7,2)
print(pairwise_distances(points1,points2))


# In[ ]:





# **TASK C: likelihood of Data Sample**

# In[ ]:


def likelihood(data,param):
    #the data has many columns (so one mean per column), two dimensional model
    d=len(data)
    p=(1/(((2*np.pi)**(d/2))*np.linalg.det(np.cov(data))))*np.exp((-1/2)*(data-param[0]).dot(np.linalg.inv(np.cov(data))).dot(data-np.mean(data)))
    return p


def best_likelihood(data,param1,param2):
    likelihoods=np.array([])
    likelihoods=np.append(likelihoods,likelihood(data,param1))
    likelihoods=np.append(likelihoods,likelihood(data,param2))
    return likelihoods.max(1)
    


# In[ ]:




