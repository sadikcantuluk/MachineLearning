# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:51:31 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('satislar.csv')

#Bağımlı ve bağımsız değişken ayırma.

aylar = veriler[['Aylar']]

satislar = veriler[['Satislar']]

#Verilerin test ve train olarak ayırılması.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

'''

#Öznitelikleme Standart Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

'''

#Lineer Regresyon model oluşturma
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

#Tahmin
tahminVerileri = lr.predict(x_test)


#Görselleştirme
x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara Göre Satışlar Grafiği")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")



























