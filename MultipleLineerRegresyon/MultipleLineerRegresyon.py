# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:57:14 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

print(veriler)

from sklearn import preprocessing

ulke = veriler.iloc[:,0:1].values

boy_kilo_yas = veriler.iloc[:,1:4].values

print(ulke)

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

#Cinsiyet Sütununun Sayısal Verilere Çevirilmesi

from sklearn import preprocessing

cinsiyet = veriler.iloc[:,4:5]

print(cinsiyet)

le = preprocessing.LabelEncoder()

cinsiyet = le.fit_transform(cinsiyet)

print(cinsiyet)


#DataFreamlerin oluşturulması ve birleştirilmesi.

sonuc1 = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
#print(sonuc1)

sonuc2 = pd.DataFrame(data=boy_kilo_yas,index=range(22),columns=['boy','kilo','yas'])
#print(sonuc2)

sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
#print(sonuc3)

sonuc4 = pd.concat([sonuc1,sonuc2],axis=1)
#print(sonuc4)

resultList = pd.concat([sonuc4,sonuc3],axis=1)
print(resultList)


#Verilerin test ve train olarak ayırılması.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(sonuc4,sonuc3,test_size=0.33,random_state=0)


#Multiple Lineer Regresyon

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

#Boyu bağımsız değişken yapma. Ornek 2

boy = sonuc2.iloc[:,0:1]
sol = resultList.iloc[:,0:3]
sag = resultList.iloc[:,4:7]

newVeri = pd.concat([sol,sag],axis=1)

x_train,x_test,y_train,y_test = train_test_split(newVeri,boy,test_size=0.33,random_state=0)

lr2 = LinearRegression()
lr2.fit(x_train,y_train)

tahmin2 = lr2.predict(x_test)


#OLS Raporu ile Geriye Eleme (Backward Elimination)

import statsmodels.api as sm

#Sabit değerimiz olmadığı için verimize birlerden oluşan bir kolon (beta0) ekliyoruz.

X = np.append(arr = np.ones((22,1)).astype(int), values = newVeri, axis=1) 

#Tüm sütunların bir listeye aktarılması.

X_list = newVeri.iloc[:,[0,1,2,3,4,5]].values

X_list = np.array(X_list, dtype = float)

rapor = sm.OLS(boy,X_list).fit()

print(rapor.summary())













