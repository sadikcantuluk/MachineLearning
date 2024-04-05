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

#Karar Ağacı Regresyonu (Decision Tree)
from sklearn.tree import DecisionTreeRegressor

dec_tree = DecisionTreeRegressor(random_state = 0)

dec_tree.fit(X,Y)

plt.scatter(X, Y, color = "green")
plt.plot(X, dec_tree.predict(X), color = "red")


print(dec_tree.predict([[11]]))
print(dec_tree.predict([[6.6]]))
print(dec_tree.predict([[2.9]]))



