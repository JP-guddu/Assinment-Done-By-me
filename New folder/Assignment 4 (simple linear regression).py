# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:38:28 2023

@author: USER
"""

#1) Delivery_time -> Predict delivery time using sorting time 
# 2) Salary_hike -> Build a prediction model for Salary_hike

#Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("C:/Users/USER/Downloads/delivery_time.csv")
df
df.info()
sns.distplot(df['Delivery Time'])
sns.distplot(df['Sorting Time'])

df=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)

df.corr()

sns.regplot(x=df['sorting_time'],y=df['delivery_time'])

model=smf.ols("delivery_time~sorting_time",data=df).fit()

#MODEL TESTING

#finding coeeficient parameter
model.tvalues , model.pvalues

#Finding Rsquare value
model.rsquared , model.rsquared_adj

#MODEL PREDICTIONS

#mannual prediction for sorting time 2
delivery_time = (6.582734) + (1.649020)*(2)
delivery_time 

#automatic prediction for say sortiing time 2,6
new_data=pd.Series([2,6])
new_data

data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred

model.predict(data_pred)
