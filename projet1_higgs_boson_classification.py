# -*- coding: utf-8 -*-
"""Projet1_Higgs-Boson-Classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HLvz5mHbnq2V4QtvI-W8reZk5eU370GE
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# %matplotlib inline 
print("Ok")

"""# Load data"""

from google.colab import drive
drive.mount('/content/drive')
#4/nQHCYB1AO4khAWQNQzkUZjsM2j5TQK6DpxfmS4OtPt8Al0zBvkUIO4o

path_test="/content/drive/My Drive/Higgs_data/test.csv"
path_train="/content/drive/My Drive/Higgs_data/train.csv"
test=pd.read_csv(path_test)
train=pd.read_csv(path_train)


#to see properly all the columns
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

print(train.head())
print(train.shape)

#from google.colab import files
#uploaded = files.upload()

'''test = pd.read_csv(io.BytesIO(uploaded['test.csv']))
train=pd.read_csv(io.BytesIO(uploaded['train.csv']))

#to see properly all the columns
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

print(train.head())
print(train.shape)
'''

"""## **Function implementation**

Lost, preparing for trainning and more
"""

def standardize(x):
  #standardize
  #transform to train with regression
  mean_x=np.mean(x)
  x-=mean_x
  std_x=np.std(x)
  x=x/std_x
  
  return x

def transform_x(x):
  tx=np.c_[np.ones(x.shape[0]),x]
  return tx

def mse(y,tx,w):
  e=y-tx.dot(w)
  mse=(1/2)*np.mean(e**2)
  return mse

def rmse(y,tx,w):
  return np.sqrt(2*mse(t,tx,w))

def build_poly(tx,degree):
  #building polynomials feature for non-linear regression, without the kernel, trick, so only for small degrees
  return phi_tx

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



"""**Training methods**"""

def least_squares(y,tx):
  a=tx.T.dot(tx)
  b=tx.T.dot(y)
  
  return np.linalg.solve(a,b)

def ridge_regression(y,tx,lambda_):
  #ridge regression for mse lost function
  a=tx.T.dot(tx)+lambda_*2*len(y)*np.identity(tx.shape[1])
  b=tx.T.dot(y)
    
  return np.linalg.solve(a,b)

"""Methods for logistic regression"""

def calculate_hessian(y, tx, w):
  S=np.zeros((y.size,y.size))
  for i in range(y.size):
    S[i][i]=sigmoid((tx[i]).T.dot(w))*(1-sigmoid((tx[i]).T.dot(w)))
  return tx.T.dot(S).dot(tx)

def sigmoid(t):
  return np.exp(t)/(1+np.exp(t))

def calculate_loss(y, tx, w):
  #loss=np.sum(np.log(np.ones(y.size)+np.exp(tx.dot(w)))-tx.dot(w).dot(y.T))
  pred = sigmoid(tx.dot(w))
  #loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
  loss=np.sum(np.log(1+np.exp(tx.dot(w))))-y.T.dot(tx.dot(w))
  return loss

  #return loss

def penalized_logistic_regression(y, tx, w, lambda_):
  loss=calculate_loss(y,tx,w)+lambda_*0.5*(w.T.dot(w))
  gradient=tx.T.dot(sigmoid(tx.dot(w))-y)+lambda_*w
  hessian=calculate_hessian(y,tx,w)+2*np.identity(w.size)
  
  return loss,gradient,hessian

def logistic_learning_by_gradient_descent(y, tx, w, gamma):
  gradient=tx.T.dot(sigmoid(tx.dot(w))-y)
  w=w-gamma*gradient
  loss=calculate_loss(y,tx,w)
  return loss, w

def logistic_learning_by_gradient_descent_penalized(y, tx, w, gamma,lambda_):
  gradient=tx.T.dot(sigmoid(tx.dot(w))-y)+lambda_*w
  w=w-gamma*gradient
  loss=calculate_loss(y,tx,w)+lambda_*0.5*(w.T.dot(w))
  return loss, w

def calculate_loss_penalized(y, tx, w):
  #loss=np.sum(np.log(np.ones(y.size)+np.exp(tx.dot(w)))-tx.dot(w).dot(y.T))
  pred = sigmoid(tx.dot(w))
  #loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
  loss=np.sum(np.log(1+np.exp(tx.dot(w))))-y.T.dot(tx.dot(w))+lambda_*0.5*(w.T.dot(w))
  return loss

def logistic_learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
  loss,gradient,hessian=penalized_logistic_regression(y,tx,w,lambda_)
  w-=gamma*((np.linalg.inv(hessian)).dot(gradient))
    
  return loss, w

def boolean_prediction_with_logistic_regression(tx,w):
  #The probability of getting 1 is by definition, definition of logistic regression, the function below. If the probability exceed 0.5 , we predict one.  
  proba_1=sigmoid(tx.dot(w))
  y_predictions=[]
  for p in proba_1:
    if p>=0.5:
      y_predictions.append(1)
    else:
      y_predictions.append(0)
  return y_predictions

def hit_mark(y_predictions,y):
  
  #Postive prediction percentage
  hit_mark=0
  for i,y_value in enumerate(y):
    if y_value==y_predictions[i]:
      hit_mark+=1
  
  return hit_mark

"""**Cross validation and hyperparameter tunning**"""

def cross_validation_ridge(data,k_indices,lambda_,degree):
  #for ridge regression
  rmse_te_temp=[] #rmse on test
  rmse_tr_temp #rmse on train set
  for k in k_indices:
    k_train=np.delete(k_indices,np.where(k_indices==k))
    train=data.iloc[k_train,:]
    test=data.iloc[k,:]
    x_train=train.iloc[:,1:]
    y_train=train.iloc[:,0]
    tx_train=transform_x(x)
    
    #getting the optimal coefficients
    w=ridge_regression(y_train,tx_train,lambda_)
    
    #Calculating rmse for this fold
    x_test=test.iloc[:,1:]
    y_test=test.iloc[:.0]
    tx_train=transform_x(x)
    
    rmse_te_temp.append(rmse(y_test,tx_test,w))
    rmse_tr_temp.append(rmse(y_train,tx_train,w))
  return np.mean(rmse_te_temp) ,np.mean(rmse_tr_temp)

def cross_validation_logistic(data,k_indices,gamma):
  
  hit_te_temp=[] #loss on test
  hit_tr_temp=[] #loss on train set
  iterations=500
  w=np.random.rand(data.shape[1])
  for k in k_indices:
    k_train=np.delete(k_indices,np.where(k_indices==k))
    train=data.iloc[k_train,:]
    test=data.iloc[k,:]
    x_train=train.iloc[:,1:]
    x_train=standardize(x_train)
    y_train=train.iloc[:,0]
    tx_train=transform_x(x)
    #getting the optimal coefficients
    for i in range(iterations):
      loss,w=logistic_learning_by_gradient_descent(y, tx, w, gamma)
    
    #Getting the test and train hit mark
    x_test=test.iloc[:,1:]
    x_test=standardize(x_test)
    y_test=test.iloc[:,0]
    tx_test=transform_x(x_test)
    y_predictions_train=sigmoid(tx_train.dot(w))
    y_predictions_test=sigmoid(tx_test.dot(w))
    
    predictions_boolean_train=[]
    for pred in y_predictions_train:
      if pred>=(0.5):
        predictions_boolean_train.append(y[1])
      else:
        predictions_boolean_train.append(y[0])
   
    predictions_boolean_test=[]
    for pred in y_predictions_test:
      if pred>=(0.5):
        predictions_boolean_test.append(y[1])
      else:
        predictions_boolean_test.append(y[0])
    
    
    hit_mark_train=hit_mark(predictions_boolean_train,y_train)
    hit_mark_test=hit_mark(predictions_boolean_test,y_test)
    
    hit_te_temp.append(hit_mark_test/y_test.shape[0])
    hit_tr_temp.append(hit_mark_train/y_train.shape[0])
    
    print("one fold")
   
  return np.mean(hit_te_temp) ,np.mean(hit_tr_temp)
 
  
  #instead of validating for a loss function, it's better to cross_validate for the hit_mark for each hyperparameter

def cross_validation_logistic_penalized(data,k_indices,gamma,lambda_):
  hit_te_temp=[] #loss on test
  hit_tr_temp=[] #loss on train set
  iterations=500
  w=np.random.rand(data.shape[1])
  for k in k_indices:
    k_train=np.delete(k_indices,np.where(k_indices==k))
    train=data.iloc[k_train,:]
    test=data.iloc[k,:]
    x_train=train.iloc[:,1:]
    x_train=standardize(x_train)
    y_train=train.iloc[:,0]
    tx_train=transform_x(x)
    #getting the optimal coefficients
    for i in range(iterations):
      loss,w=logistic_learning_by_gradient_descent_penalized(y, tx, w, gamma,lambda_)
    
    #Getting the test and train hit mark
    x_test=test.iloc[:,1:]
    x_test=standardize(x_test)
    y_test=test.iloc[:,0]
    tx_test=transform_x(x_test)
    y_predictions_train=sigmoid(tx_train.dot(w))
    y_predictions_test=sigmoid(tx_test.dot(w))
    
    predictions_boolean_train=[]
    for pred in y_predictions_train:
      if pred>=(0.5):
        predictions_boolean_train.append(y[1])
      else:
        predictions_boolean_train.append(y[0])
   
    predictions_boolean_test=[]
    for pred in y_predictions_test:
      if pred>=(0.5):
        predictions_boolean_test.append(y[1])
      else:
        predictions_boolean_test.append(y[0])
    
    
    hit_mark_train=hit_mark(predictions_boolean_train,y_train)
    hit_mark_test=hit_mark(predictions_boolean_test,y_test)
    
    hit_te_temp.append(hit_mark_test/y_test.shape[0])
    hit_tr_temp.append(hit_mark_train/y_train.shape[0])
    
    print("one fold")
   
  return np.mean(hit_te_temp) ,np.mean(hit_tr_temp)
 
  
  #instead of validating for a loss function, it's better to cross_validate for the hit_mark for each hyperparameter



"""## **Data Exploration and feature engineering**"""

train=train.drop('Id',axis=1)

train=train.replace(-999.000,np.nan)
train=train.fillna(train.mean())
#Replacing the -999 by the average, since it would be too much to drop everything

train.head()

"""**Load data**"""

y=train.iloc[:,0]
x=train.iloc[:,1:]
x=standardize(x)
tx=transform_x(x)
tx[:5,:]

"""**Try predicting with logistic regression we define b as 1 and s as 0**"""

tx.shape

y[y=='b']=1
y[y=='s']=0
print(y[:20])
#tx was probably as float object instead of float64
y=y.astype(float)
tx=tx.astype(float)

losses=[]
w=np.random.rand(tx.shape[1])
print(w)
w=w.astype(float)
lambda_=0
gamma=0.000001
iterations=500
for i in range(iterations):
  print("iteration ",i+1)
  loss,w=logistic_learning_by_gradient_descent(y, tx, w, gamma)
  losses.append(loss)
losses.append(calculate_loss(y,tx,w))
print(losses)
plt.plot(range(iterations+1),losses)

print(tx.dot(w))
y_predictions=sigmoid(tx.dot(w))
print(y_predictions)

predictions_boolean=[]
for pred in y_predictions:
  if pred>=(0.5):
    predictions_boolean.append(y[1])
  else:
    predictions_boolean.append(y[0])
hit=hit_mark(predictions_boolean,y)

print(hit/len(predictions_boolean))

"""**As we can see, the prediction with logistic regression is pretty good with 75% correct prediction**

We can now try cross validating to see how it performs on a test set
"""

k_indices=build_k_indices(y,5,1)
gamma=0.000001
hit_mark_test,hit_mark_train=cross_validation_logistic(train,k_indices,gamma)
print(" the hit mark for train and test respectively: ")
print(hit_mark_test)
print(hit_mark_train)

"""**Now with penalized logistic regression**"""

losses=[]
w=np.random.rand(tx.shape[1])
print(w)
w=w.astype(float)
lambda_=1
gamma=0.000001
iterations=100
for i in range(iterations):
  print("iteration ",i+1)
  loss,w=logistic_learning_by_gradient_descent_penalized(y, tx, w, gamma,lambda_)
  losses.append(loss)
losses.append(calculate_loss_penalized(y,tx,w))
print(losses)
plt.plot(range(iterations+1),losses)

k_indices=build_k_indices(y,5,1)
gamma=0.000001
lambda_=1
hit_mark_test,hit_mark_train=cross_validation_logistic_penalized(train,k_indices,gamma,lambda_)
print(" the hit mark for train and test respectively: ")
print(hit_mark_test)
print(hit_mark_train)



