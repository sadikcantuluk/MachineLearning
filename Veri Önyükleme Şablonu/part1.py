# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:59:19 2024

@author: sadik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Verinin Okunması

veriler = pd.read_csv('veriler.csv')
print(veriler)

boy = veriler[['boy']]

boy_kilo = veriler[['boy','kilo']]

print(boy_kilo)

#Temel Python 

class Car:
     model = "Symbol"
     def Start(self):
         print("Araba çalıştı.")
         
araba = Car()
araba.Start()





         
         

    
