# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:45:40 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

hacim_maas = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmean = KMeans(n_clusters = 3, init = 'k-means++')

# init başlangıç küme merkezlerinin nasıl seçileceğini belirler.
# "random" (varsayılan) diğer seçenek 'k-means++' dir.

kmean.fit(hacim_maas)

print(kmean.cluster_centers_)

# WCSS Değeri ile En Uygun n_clusters Sayısını Bulma

WCSS_Degerleri = []

for i in range(1, 11):
    kmean2 = KMeans(n_clusters = i, init = 'k-means++')
    kmean2.fit(hacim_maas)
    WCSS_Degerleri.append(kmean2.inertia_) #inertia_ WCSS değerini verir.
    
print(WCSS_Degerleri)
plt.plot(range(1, 11), WCSS_Degerleri)












