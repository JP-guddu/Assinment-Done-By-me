# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:36:01 2023

@author: USER
"""

import pandas as pd
from scipy import stats 
import numpy as np

buyeratio= pd.read_csv("C:/Users/USER/Downloads/BuyerRatio.csv")
buyeratio
buyeratio_table = buyeratio.iloc[:,1:6]
buyeratio_table
buyeratio_table.values
val=stats.chi2_contingency(buyeratio_table)
val
