# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:49:29 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")

from sklearn.impute import SimpleImputer

boy_kilo_yas = veriler.iloc[:,1:4].values

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

veriler.iloc[:,1:4] = imputer.fit_transform(boy_kilo_yas)

#Kategorik veriler

from sklearn import preprocessing

ulke = veriler.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

le.fit_transform(veriler.iloc[:,0:1])

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()



s1 = pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(s1)

s2 = pd.DataFrame(data=boy_kilo_yas,index=range(22),columns=['boy','kilo','yas'])
print(s2)

cinsiyet = veriler.iloc[:,4:5].values
s3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(s3)

s = pd.concat([s1,s2],axis=1)
print(s)

result = pd.concat([s,s3],axis=1)
print(result)


























