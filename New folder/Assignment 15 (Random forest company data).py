# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:17:32 2023

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

company = pd.read_csv("C:/Users/USER/Downloads/Company_Data (1).csv")
company
company.head()
company.describe()
company.dtypes
company.info()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

company["ShelveLoc"] = label_encoder.fit_transform(company["ShelveLoc"])
company["Urban"] = label_encoder.fit_transform(company["Urban"])
company["US"] = label_encoder.fit_transform(company["US"])

company

feature_cols=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
company['High'] = company.Sales.map(lambda x: 1 if x>=8 else 0)
company

x = company.drop(['Sales', 'High'], axis = 1)
x = company[feature_cols]
y = company.High

print(x)
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 40)
print(x_train)
print(y_train)
print(y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


#Training the Random Forest Classification model on the Training data
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 40)
classifier.fit(x_train, y_train)

classifier.fit(x_train, y_train)
classifier.score(x_test, y_test)

#Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)
accuracy_score(y_test, y_pred)

classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
classifier.fit(x_train, y_train)

classifier.score(x_test, y_test)


