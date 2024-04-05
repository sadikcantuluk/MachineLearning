# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:11:37 2024

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

#Lineer Regresyon

from sklearn.linear_model import LinearRegression

lineerReg = LinearRegression()

lineerReg.fit(X, Y)

plt.scatter(X, Y, color = "red")
plt.plot(X,lineerReg.predict(X), color = "blue")
         

from sklearn.preprocessing import PolynomialFeatures

polinomReg = PolynomialFeatures(degree = 4)

X_pol = polinomReg.fit_transform(X)

lineerReg2 = LinearRegression()

lineerReg2.fit(X_pol, Y)

plt.scatter(X,Y)
plt.plot(X,lineerReg2.predict(polinomReg.fit_transform(X)))


#Tahminler

print(lineerReg.predict([[6.6]]))
print(lineerReg.predict([[11]]))

print("---------------------------")

print(lineerReg2.predict(polinomReg.fit_transform([[6.6]])))
print(lineerReg2.predict(polinomReg.fit_transform([[11]])))