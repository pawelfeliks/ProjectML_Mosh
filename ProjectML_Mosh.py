#!/usr/bin/env python
# coding: utf-8

# In[29]:


# import all frameworks needed
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
# import joblib


# In[3]:


# import dataset file
df = pd.read_csv('music.csv')
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[ ]:


# data parts for ML model (X,y)


# In[7]:


X = df.drop(columns=['genre'])
X.head()


# In[8]:


y = df['genre']
y.head()


# In[16]:


# split data to train, test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[20]:


# definition of the model
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

# # first prediction for age 21 and gender male (1) and 22 female (0)
# predictions = model.predict([[21,1],[22,0]])

# prediction form train and test datasets
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
score


# In[30]:


# visualization through "dot" file
tree.export_graphviz(model, out_file='music_recommender.dot',
                    feature_names=['age', 'gender'],
                    class_names=sorted(y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)


# In[ ]:




