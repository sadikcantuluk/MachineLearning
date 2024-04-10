# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:08:25 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar_yeni.csv')

unvan = veriler.iloc[:,1:2]
unvan_seviyesi = veriler.iloc[:,2:3]
kidem = veriler.iloc[:,3:4]
puan = veriler.iloc[:,4:5]
maas = veriler.iloc[:,5:6]

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

veriler.iloc[:,1:2] = le.fit_transform(veriler.iloc[:,1:2])

# Dataframes

unvanSeviyesi_kidem = pd.concat([unvan_seviyesi,kidem], axis = 1)
unvanSeviyesi_kidem_puan = pd.concat([unvanSeviyesi_kidem,puan], axis = 1)

# Eğitim

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(unvanSeviyesi_kidem_puan,maas,test_size = 0.33, random_state = 0)

# Lineer Rgresyon

from sklearn.linear_model import LinearRegression

multi_linReg = LinearRegression()
multi_linReg.fit(x_train, y_train)



print(multi_linReg.predict(x_test))


























