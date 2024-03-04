# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:39:56 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri okuma
satisVerileri = pd.read_csv('satislar.csv')

#bağımlı ve bağımsız değişkenleri vermek için ayırma işlemi

aylar = satisVerileri.iloc[:,0:1].values
aylar = pd.DataFrame(data=aylar,index=range(30),columns=['Aylar'])
satislar = satisVerileri.iloc[:,1:2].values
satislar = pd.DataFrame(data=satislar,index=range(30),columns=['Satislar'])

#### Verilerin Eğitim ve Test Verisi Olarak Ayırma

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

'''
#### Öznitelik Ölçekleme

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

'''
####model inşası (linear regression)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

###tahmin işlemi
tahminVerileri = lr.predict(x_test) 

####görselleştirme

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))














