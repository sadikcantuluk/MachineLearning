# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:29:58 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri okuma
veriler = pd.read_csv('veriler.csv')

boy_kilo_yas = veriler.iloc[:,1:4].values

cinsiyet = veriler.iloc[:,4:5].values

#### Verilerin Eğitim ve Test Verisi Olarak Ayırma

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(boy_kilo_yas,cinsiyet,test_size=0.33,random_state=0)

#### Öznitelik Ölçekleme

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test) 


#Logistic Regression 

from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(random_state = 0)

lgr.fit(x_train, y_train)

y_pred = lgr.predict(x_test)

print(y_pred)


# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)




















 