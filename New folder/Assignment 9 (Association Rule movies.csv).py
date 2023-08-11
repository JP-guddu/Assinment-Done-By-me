# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:03:00 2023

@author: USER
"""

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


movie=pd.read_csv("C:/Users/USER/Downloads/my_movies.csv")
movie

movie.shape

movie.info()
movie.corr()

movie2=movie.iloc[:,5:]
movie2

#Apriori Algorithm
#Association rules with 10% Support and 70% confidence

# With 10% Support
frequent_itemsets = apriori(movie2, min_support=0.1, use_colnames=True)
frequent_itemsets

# with 70% confidence
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules

#####  An leverage value of 0 indicates independence. Range will be [-1 1]
# high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]

#Sorting Value By Decending Order

rules.sort_values('lift',ascending = False)[0:20]

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]

# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# Association rules with 5% Support and 90% Confidence

# With 5% Support
frequent_itemsets = apriori(movie2, min_support=0.05, use_colnames=True)
frequent_itemsets

# with 90% confidence
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.9)
rules

rules.sort_values('lift',ascending = False)[0:20]

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]


# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()



































