#!/usr/bin/env python
# coding: utf-8

# 訓練 😉

# 載入函式庫


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import data_generator


# 讀入資料

df = pd.read_csv("data/student_data_30000.csv")
print(df.head())


# 特徵

features = df.drop("Grade", axis=1)
features = pd.get_dummies(features, drop_first=True)
print(features.head())


# 類別

label = df["Grade"]
print(label.head())


# 將資料分成訓練集(0.8)與測試集(0.2)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.20)


# Decision Tree 🌲

clf = tree.DecisionTreeClassifier(min_samples_split=2000, min_samples_leaf=1000, max_depth=8)
clf.fit(X_train, y_train)
pred_label = clf.predict(X_test)
print(accuracy_score(y_test, pred_label))


# Support Vector Machine 🗿

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
pred_label = clf.predict(X_test)
accuracy_score(y_test, pred_label)
