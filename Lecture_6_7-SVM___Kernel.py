#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

print("ok")


# In[ ]:


url="https://raw.githubusercontent.com/epfml/ML_course/master/labs/ex07/template/data/train.csv"
train=pd.read_csv(url)

#to see properly all the columns
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

train.head


# 
# # Helper functions
# ---
# 
# 

# In[ ]:


def load_prepare_data(data):
  y=data.iloc[:,1]
  x=data.iloc[:,2:]
  
  #transform b in -1 and s in 1
  
  yb=np.ones(y.shape[0])
  yb[np.where(y=='b')]=-1
  
  return yb,x
  


# In[ ]:


def standardize(train):
  mean_train=np.mean(train)
  std_train=np.std(train)
  
  train-=mean_train
  train=train/std_train
  
  return train
  
 


# In[ ]:


def predict_labels(weights,data):
  predictions=data.dot(weights)
  predictions[np.where(predictions<=0)]=-1
  predictions[np.where(predictions>0)]=1
  
  return predictions


# In[ ]:


def transform_x(x):
  tx=np.c_[np.ones(x.shape[0]),x]
  return tx


# In[ ]:


def calculate_accuracy(y,tx,w):
  predictions=predict_labels(w,tx)
  print(predictions)
  hit_mark=0
  for i,pred in enumerate(predictions):
    if pred==y[i]:
      hit_mark+=1
  
  return hit_mark/y.shape[0]


# In[ ]:


def calculate_primal_objective(y,tx,w,lambda_):
  #calculate the loss of the real loss function, this might not be as useful as the accuracy for a boolean classification
  elements=np.maximum(np.zeros(y.shape[0]),1-y*(np.dot(tx,w)))
  loss=np.sum(elements)+lambda_*(np.dot(w,w))*0.5
  
  return loss


# **Exercice 1: Stochastic Gradient Descent for SVM (Ridge)**

# In[ ]:


def calculate_stochastic_gradient(y,tx,w,lambda_,n):
  #n represents a random n for element wise stochastic gradient descent
  N=tx.shape[0]
  x_n,y_n=tx[n],y[n]
  if (1-y_n*(x_n.T).dot(w))>0:
    return -y_n*x_n+lambda_*w
  else:
    return 0+lambda_*w
  


# In[ ]:


def sgd_for_svm_demo(y,tx):
  max_iter= 100000
  gamma=0.001
  lambda_=0.1

  w=np.random.rand(tx.shape[1])
  for i in range(max_iter):
    n=np.random.randint(0,tx.shape[0])
    gradient=calculate_stochastic_gradient(y,tx,w,lambda_,n)
    w-=gamma*gradient
    
    if i%10000==0:
      print("iteration={it}, cost={c}".format(it=i,c=calculate_primal_objective(y,tx,w,lambda_)))
    
  print("trainning accuracy is, for the last loss,  ", calculate_accuracy(y,tx,w)*100, "%")
 
    


# In[ ]:


y,x=load_prepare_data(train)
x=standardize(x)
tx=transform_x(x)

index = np.random.choice(tx.shape[0], 250, replace=False)  
tx=tx[index]
y=y[index]
               


# In[ ]:


sgd_for_svm_demo(y,tx)


# Prediction seems highly variant without ridge, on the other hand with ridge it's way better

# # **Coordinate descent for dual representation of svm**

# In[ ]:


def calculate_coordinate_update(y,X,lambda_,alpha,w,n):
  #n represents the randomly chosen coordinate. For every exemple there's one alpha
  x_n,y_n=X[n],y[n]
  old_alpha_n=np.copy(alpha[n])
  g=y_n*x_n.dot(w)-1
  if old_alpha_n==0:
    g=min(0,g)
  elif old_alpha_n==1.0:
    g=max(0,g)
  else:
    g=g
  if g!=0:
    alpha[n]=min(max(old_alpha_n- lambda_ * g / (x_n.T.dot(x_n)),0),1)
    
    w+=1.0/lambda_*(alpha[n]-old_alpha_n)*x_n*y_n
  return w,alpha
  


# In[ ]:


def calculate_dual_objective(y,X,w,alpha,lambda_):
  return np.sum(alpha)- lambda_ / 2.0 * np.sum(w ** 2)


# In[ ]:


def coordinate_descent_for_svm_demo(y,X):
  max_iter=100000
  lambda_=0.001
  w=np.zeros(X.shape[1])
  alpha=np.zeros(X.shape[0])
  
  for it in range(max_iter):
    n=np.random.randint(0,X.shape[0])
    w,alpha=calculate_coordinate_update(y,X,lambda_,alpha,w,n)
    if it % 10000 == 0:
      # primal objective
      primal_value = calculate_primal_objective(y, X, w, lambda_)
      # dual objective
      dual_value = calculate_dual_objective(y, X, w, alpha, lambda_)
      # primal dual gap
      duality_gap = primal_value - dual_value
      print('iteration=%i, primal:%.5f, dual:%.5f, gap:%.5f'%(
                    it, primal_value, dual_value, duality_gap))
  print("training accuracy = {l}".format(l=calculate_accuracy(y, X, w)))

coordinate_descent_for_svm_demo(y, tx)


#  **Extension: trying Kernel trick with svm**

# In[ ]:


def kernel_function(xi,xj,gamma):
  return (1+xj.T.dot(xi))*3
  #return np.exp(-gamma*((xi-xj).dot((xi-xj))))
  


# In[ ]:


def calculate_kernel_matrix(X,gamma):
  matrix=np.zeros((X.shape[0],X.shape[0]))
  for i in range(X.shape[0]):
    for j in range(X.shape[0]):
      matrix[i][j]=kernel_function(X[i],X[j],gamma)
  print("matrix",matrix)
  return matrix
      


# In[ ]:


def predict_label_kernel(alphas,kernel_matrix):
  predictions=kernel_matrix.dot(alphas)
  
  print("hello",predictions)
  predictions[np.where(predictions<=0)]=-1
  predictions[np.where(predictions>0)]=1
  
  return predictions
  
  


# In[ ]:


def calculate_Q_matrix(y,kernel_matrix):
  return y*kernel_matrix*y


# In[ ]:


def calculate_dual_loss_kernel(Q,lambda_,alpha):
  return np.sum(alpha)-0.5*lambda_*alpha.T.dot(Q).dot(alpha)

  


# In[ ]:


def calculate_coordinate_update_kernel(y,Q,lambda_,alpha,n):
  sum_=0
  for i in range(Q.shape[0]):
    if i!=n:
      sum_+=Q[n][i]*alpha[i]
  
  alpha[n]=1/Q[n][n]*(lambda_-sum_)
  
  return alpha
 
  
  


# In[ ]:


def coordinate_descent_for_svm_kernel_demo(y,X):
  gamma=0.001
  kernel_matrix=calculate_kernel_matrix(X,gamma)
  Q=calculate_Q_matrix(y,kernel_matrix)
  
  max_iter=100000
  lambda_=1
  alpha=np.zeros(X.shape[0])
  
  for it in range(max_iter):
    n=np.random.randint(0,X.shape[0])
    alpha=calculate_coordinate_update_kernel(y,Q,lambda_,alpha,n)
    
    if it%10000==0: 
      print(calculate_dual_loss_kernel(Q,lambda_,alpha))
  print("alpha",alpha)
  print(calculate_accuracy_kernel(y,kernel_matrix,alpha))
    
  
    
    
  


# In[ ]:


def calculate_accuracy_kernel(y,kernel_matrix,alpha):
  predictions=predict_label_kernel(alpha,kernel_matrix)
  print(predictions)
  hit_mark=0
  for i,pred in enumerate(predictions):
    if pred==y[i]:
      hit_mark+=1
  
  return hit_mark/y.shape[0]
  


# In[ ]:


print(tx.shape)


# In[ ]:


coordinate_descent_for_svm_kernel_demo(y,tx)


# In[ ]:




