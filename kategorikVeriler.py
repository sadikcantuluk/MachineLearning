# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:09:52 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()


print(ulke)