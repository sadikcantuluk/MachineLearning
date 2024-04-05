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

#Rassal Ağaç ile Tahmin (Random Forest)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0) 

# n_estimators kaç tane karar ağacı oluşturulacağını belirler.

rf_reg.fit(X, Y)

plt.scatter(X, Y, color = "red")
plt.plot(X, rf_reg.predict(X), color = "blue")


print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))
print(rf_reg.predict([[2.9]]))



