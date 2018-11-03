#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import data_generator


# In[3]:


df = pd.read_csv("student_data_30000.csv")
df.head()


# In[4]:


features = df.drop("Grade", axis=1)
features = pd.get_dummies(features, drop_first=True)
features.head()


# In[5]:


label = df["Grade"]
label.head()


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.20)


# In[7]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[8]:


pred_label = clf.predict(X_test)


# In[9]:


accuracy_score(y_test, pred_label)


# In[10]:


clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)


# In[11]:


pred_label = clf.predict(X_test)


# In[12]:


accuracy_score(y_test, pred_label)


# In[13]:


# !jupyter nbconvert --to script training.ipynb

