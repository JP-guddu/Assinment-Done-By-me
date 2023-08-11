# -*- coding: utf-8 -*-,,
"""
Created on Mon Mar 20 10:05:38 2023

@author: USER
""",,

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
#conda install -c conda-forge mlxtend
!pip install mlxtend

book= pd.read_csv("C:/Users/USER/Downloads/book (1).csv")
book
book.head()
book.shape
book.dtypes
book.corr()

#As the data is not in transaction formation We are using transaction Encoder

df=pd.get_dummies(book)
df
df.head()

#Apriori Algorithm
#Association rules with 10% Support and 70% confidence

# With 10% Support
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

# with 70% confidence
rules = association_rules(frequent_itemsets,metric="lift", min_threshold=0.7)
rules

#Sorting Value By Decending Order
rules.sort_values('lift',ascending = False)[0:20]

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions,,
rules[rules.lift>1]

# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

# Association rules with 15% Support and 85% confidence
# With 15% Support
frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True)
frequent_itemsets

# with 80% confidence
rules = association_rules(frequent_itemsets,metric= "lift", min_threshold=0.85)
rules

rules.sort_values('lift',ascending = False)[0:20]


# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

#Association rules for Support 5% & confidence 90%
# With 5% Support
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
frequent_itemsets

# with 80% confidence,,
rules = association_rules(frequent_itemsets," metric=""lift""", min_threshold=0.9)
rules

rules.sort_values('lift',ascending = False)[0:20]

# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()
