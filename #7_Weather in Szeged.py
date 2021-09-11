import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import csv
import torch
import warnings
warnings.filterwarnings('ignore')

data_path= '/Users/zafarzhonirismetov/Desktop/weatherHistory.csv'

data = pd.read_csv(data_path)
data.drop("Loud Cover" , axis = 1,inplace = True)
data['Pressure (millibars)'].replace(0,np.nan, inplace =True)
data.fillna(method = 'pad',inplace = True)
data['Formatted Date'] = pd.to_datetime(data['Formatted Date'].apply(lambda x: x.split('+')[0]) )
data = data.set_index('Formatted Date')
data = data.sort_index()

plt.plot(data['Humidity'],data['Temperature (C)'])
plt.show()