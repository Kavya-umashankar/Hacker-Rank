#!/usr/bin/env python
# coding: utf-8

# # Laptop Battery life
# 
# Fred is a very predictable man. For instance, when he uses his laptop, all he does is watch TV shows. He keeps on watching TV shows until his battery dies. Also, he is a very meticulous man, i.e. he pays great attention to minute details. He has been keeping logs of every time he charged his laptop, which includes how long he charged his laptop for and after that how long was he able to watch the TV. Now, Fred wants to use this log to predict how long will he be able to watch TV for when he starts so that he can plan his activities after watching his TV shows accordingly.
# 
# Challenge
# 
# You are given access to Fred’s laptop charging log by reading from the file “trainingdata.txt”. The training data file will consist of 100 lines, each with 2 comma-separated numbers.
# 
# The first number denotes the amount of time the laptop was charged.
# The second number denotes the amount of time the battery lasted.
# The training data file can be downloaded here (this will be the same training data used when your program is run). The input for each of the test cases will consist of exactly 1 number rounded to 2 decimal places. For each input, output 1 number: the amount of time you predict his battery will last.
# 
# Sample Input
# 
# 1.50
# 
# Sample Output
# 
# 3.00
# 
# Scoring
# 
# Your score will be 10-X, where X is the sum of the distances you are from expected answer of each test case. For instance if there are 2 test cases with expected answer 4 and you print 3 for the first one and 6 for the second one your score will be 10-(1+2) = 7.

# In[37]:


import math
import os
import random
import re
import sys


# In[38]:


import pandas as pd


# In[55]:


file = open('laptopcharging.txt','r')
lst=[]
for line in file:
    temp = line.replace('\n','')
    lst += [temp.split(",")]
print(lst)


# In[56]:


df = pd.DataFrame(lst,columns=['chargingtime','dischargingtime'])
df.shape


# In[57]:


df = df.astype({"chargingtime":"float","dischargingtime":"float"})
df


# In[58]:


#data visulization
import matplotlib.pyplot as plt
plt.plot(df.iloc[:,0], df.iloc[:,1], 'ro')
plt.ylabel('Laptob Battery Life')
plt.show()


# In[59]:


df = df[df.iloc[:,1] < 8]


# In[60]:


#ADD BAISSSSSSSSSSSSSSSSSSSS
df.insert(0, len(df.columns), 0)


# In[65]:


df


# In[74]:


X = df.iloc[:,0:2].values
Y = df.iloc[:,2].values


# In[85]:


X


# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,Y ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)


# In[78]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[79]:


y_pred = model.predict(X_test)


# In[80]:


from sklearn import metrics
metrics.mean_absolute_error(y_test,y_pred)


# In[81]:


metrics.mean_squared_error(y_test,y_pred)


# In[82]:


np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# In[86]:


model.predict([[0,0.09]])


# In[ ]:




