# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:05:12 2023

@author: USER
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm

labtat = pd.read_csv("C:/Users/USER/Downloads/LabTAT.csv")
labtat

#useing Anova Ftest statistic
p_value = stats.f_oneway(labtat.iloc[:,0],labtat.iloc[:,1],labtat.iloc[:,2],labtat.iloc[:,3])
p_value
