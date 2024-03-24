# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:02:09 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri okuma
veriler = pd.read_csv('eksikveriler.csv')

#eksik verileri ortlama değere göre doldurma.
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

boy_kilo_yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(boy_kilo_yas)

boy_kilo_yas = imputer.transform(boy_kilo_yas)

#VEYA TEK SATIRDA boy_kilo_yas = imputer.fit_transform(boy_kilo_yas)

print(boy_kilo_yas)

'''

Bu Python kodu, eksik verilerin bulunduğu bir CSV dosyasını okuyarak, eksik verileri ortalama değerlerle doldurur. Ardından, eksik verilerin doldurulmuş haliyle 
bir dizi işlem yapar ve sonuçları ekrana yazdırır. Şimdi kodu adım adım açıklayalım:

1. `import pandas as pd`: Bu satır, pandas kütüphanesini `pd` kısaltmasıyla içe aktarır. Pandas, veri manipülasyonu ve analizi için yaygın olarak kullanılan bir 
Python kütüphanesidir.

2. `import numpy as np`: Bu satır, NumPy kütüphanesini `np` kısaltmasıyla içe aktarır. NumPy, çok boyutlu dizilerle çalışmak için kullanılan bir Python kütüphanesidir.

3. `import matplotlib.pyplot as plt`: Bu satır, matplotlib kütüphanesinden `pyplot` modülünü `plt` kısaltmasıyla içe aktarır. Matplotlib, veri görselleştirme 
için yaygın olarak kullanılan bir Python kütüphanesidir.

4. `veriler = pd.read_csv('eksikveriler.csv')`: Bu satır, `'eksikveriler.csv'` dosyasından verileri bir DataFrame'e yükler. Pandas'ın `read_csv()` fonksiyonu 
CSV dosyasını okuyarak verileri bir DataFrame'e dönüştürür.

5. `from sklearn.impute import SimpleImputer`: Bu satır, scikit-learn kütüphanesinden `SimpleImputer` sınıfını içe aktarır. Scikit-learn, Python'da kullanılan 
popüler bir makine öğrenimi kütüphanesidir.

6. `imputer = SimpleImputer(missing_values=np.nan, strategy='mean')`: Bu satır, eksik değerleri doldurmak için bir `SimpleImputer` nesnesi oluşturur. `missing_values` 
parametresi eksik değerlerin nasıl tanımlanacağını belirtir (burada NaN, yani "Not a Number" kullanılır). `strategy` parametresi ise eksik değerlerin nasıl 
doldurulacağını belirtir (burada ortalama değer kullanılır).

7. `boy_kilo_yas = veriler.iloc[:,1:4].values`: Bu satır, DataFrame'den yalnızca boy, kilo ve yaş sütunlarını seçerek bir NumPy dizisine dönüştürür. `iloc[]` 
kullanarak DataFrame'de belirli bir konumda bulunan verilere erişilir.

8. `imputer = imputer.fit(boy_kilo_yas)`: Bu satır, `imputer` nesnesini eğitir. Eğitim işlemi, eksik değerleri doldurmak için kullanılacak istatistikleri hesaplar 
(burada sadece ortalama değer hesaplanır).

9. `boy_kilo_yas = imputer.transform(boy_kilo_yas)`: Bu satır, eksik değerleri doldurarak `boy_kilo_yas` dizisini değiştirir. `transform()` yöntemi, eksik değerleri
doldurmak için önceden eğitilmiş `imputer` nesnesini kullanır.

10. `print(boy_kilo_yas)`: Son olarak, doldurulmuş verileri ekrana yazdırır.

Bu kod özetle, eksik verileri ortalama değerlerle doldurarak veri setini işler. Bu, eksik verilerin makine öğrenimi modelleri tarafından daha iyi işlenebilmesini sağlar.

'''

