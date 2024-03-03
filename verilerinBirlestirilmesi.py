# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:34:59 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri okuma
eksikveriler = pd.read_csv('eksikveriler.csv')

#eksik verileri ortlama değere göre doldurma.
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

boy_kilo_yas = eksikveriler.iloc[:,1:4].values

imputer = imputer.fit(boy_kilo_yas)

boy_kilo_yas = imputer.transform(boy_kilo_yas)


veriler = pd.read_csv('veriler.csv')

ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()


#DataFreamlerin oluşturulması ve birleştirilmesi.


sonuc1 = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
#print(sonuc1)

sonuc2 = pd.DataFrame(data=boy_kilo_yas,index=range(22),columns=['boy','kilo','yas'])
#print(sonuc2)

cinsiyet = veriler.iloc[:,-1]
#print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
#print(sonuc3)

sonuc4 = pd.concat([sonuc1,sonuc2],axis=1)
#print(sonuc4)

resultList = pd.concat([sonuc4,sonuc3],axis=1)
print(resultList)

#### Verilerin Eğitim ve Test Verisi Olarak Ayırma

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(sonuc4,cinsiyet,train_size=0.8,random_state=0)

#### Öznitelik Ölçekleme

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



