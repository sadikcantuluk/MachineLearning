# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:53:34 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri okuma
df = pd.read_csv('arac_verileri.csv')

x = df.iloc[:,1:7]
y = df.iloc[:0:1]

X = x.replace({',','.'},regex=True)
Y = x.replace({',','.'},regex=True)
#replace metodunu kullanarak verilerdeki virgülleri noktaya çevir.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_pred)

print("R-Scuared",r2*100)

#veri okuma
##Doğrusal , Polinom , Kara Ağacı ,Random Forres Regresyon,