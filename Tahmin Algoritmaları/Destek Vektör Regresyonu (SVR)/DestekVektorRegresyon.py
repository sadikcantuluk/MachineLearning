# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:06:24 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri yükleme

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:3]

X = x.values
Y = y.values

#Standart Scaler Uygulama
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()

X_olcekli = sc1.fit_transform(X)
Y_olcekli = sc2.fit_transform(Y)

#Destek Vektör Regresyonu (SVR)
from sklearn.svm import SVR

svrRegresyon = SVR(kernel = "rbf")
svrRegresyon.fit(X_olcekli, Y_olcekli)

plt.scatter(X_olcekli, Y_olcekli, color = "red")
plt.plot(X_olcekli, svrRegresyon.predict(X_olcekli), color = "blue")


print(svrRegresyon.predict([[11]]))


