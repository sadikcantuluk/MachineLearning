# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:56:34 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('odev_tenis.csv')

from sklearn import preprocessing

outlook = veriler.iloc[:,0:1].values
windy = veriler.iloc[:,3:4].values
play = veriler.iloc[:,4:5].values
temperature = veriler.iloc[:,1:2].values
humidity = veriler.iloc[:,2:3].values

le = preprocessing.LabelEncoder()

windy = le.fit_transform(veriler.iloc[:,3:4])

play = le.fit_transform(veriler.iloc[:,4:5])

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

outlookDf = pd.DataFrame(data=outlook,index=range(14),columns=['overcast','rainy','sunny'])
temperature = pd.DataFrame(data=temperature,index=range(14),columns=['temperature'])
windy = pd.DataFrame(data=windy,index=range(14),columns=['windy'])
play = pd.DataFrame(data=play,index=range(14),columns=['play'])
humidity = pd.DataFrame(data=humidity,index=range(14),columns=['humidity'])

outlookDf_temperature = pd.concat([outlookDf,temperature],axis=1)

windy_play = pd.concat([windy,play],axis=1)

result = pd.concat([outlookDf_temperature,windy_play],axis=1)

print(result)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(result,humidity,test_size=0.33,random_state=0)

#Multiple Lineer Regresyon

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

x_train = x_train.sort_index()
y_train = y_train.sort_index()

lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)


#OLS 
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = result, axis = 1)

X_List = result.iloc[:,[0,1,2,3,4,5]].values
X_List = np.array(X_List,dtype = float)

rapor = sm.OLS(humidity,result).fit()

print(rapor.summary())


























