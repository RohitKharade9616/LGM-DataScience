#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import required libraries 
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\Admin\iris-1.csv")
df.head()


# In[10]:


# basic information of dataset
print(df.shape)
print(df.columns)
print(df.dtypes)


# In[6]:


# descriptive statistics
print(df.describe())


# In[11]:


# correlation between variables
co=df.corr()
sns.heatmap(data=co)
plt.show()


# In[12]:


# drop a "Id" column because it cannot required further
df=df.drop("Id",axis=1)
df.head()


# In[13]:


# detecting missing values in a dataset
print(df.isnull().sum())


# In[14]:


x=df.iloc[:,:4]
y=df.iloc[:,4]


# In[15]:


# splitting a dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[17]:


# model building
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[18]:


# model testing
y_pred=model.predict(x_test)
y_pred


# In[21]:


# model evaluation
print(model.score(x_test,y_test))


# In[22]:


from sklearn.metrics import *
print(classification_report(y_test,y_pred))


# In[23]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# In[24]:


# visualize decision tree
from sklearn import tree
tree.plot_tree(model)
plt.show()


# In[ ]:




