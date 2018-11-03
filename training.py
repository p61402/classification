#!/usr/bin/env python
# coding: utf-8

# In[512]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import data_generator


# In[513]:


df = pd.read_csv("training_data_30000.csv")
df.head()


# In[514]:


features = df.drop("Grade", axis=1)
features = pd.get_dummies(features, drop_first=True)
features.head()


# In[515]:


label = df["Grade"]
label.head()


# In[516]:


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.20)


# In[517]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[518]:


pred_label = clf.predict(X_test)


# In[519]:


accuracy_score(y_test, pred_label)


# In[520]:


clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)


# In[521]:


pred_label = clf.predict(X_test)


# In[522]:


accuracy_score(y_test, pred_label)


# In[524]:


# !jupyter nbconvert --to script training.ipynb

