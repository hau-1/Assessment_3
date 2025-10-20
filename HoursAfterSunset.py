import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np



ds1 = pd.read_csv('dataset1.csv')

x = pd.to_numeric(ds1['hours_after_sunset'], errors='coerce')
y = pd.to_numeric(ds1['bat_landing_to_food'], errors='coerce')

yMean = y.mean()    
ySTD = y.std()

xMean = x.mean()
xSTD = x.std()

#filter outliers
mask = x.between(xMean - 2*xSTD, xMean + 2*xSTD) & y.between(yMean - 2*ySTD, yMean + 2*ySTD)
x_filtered = x[mask]
y_filtered = y[mask]

plt.scatter (x_filtered, y_filtered, color='tab:blue', alpha=0.5)
plt.title('Hours After Sunset vs Bat Landing Time to Food')
plt.xlabel('Hours After Sunset')
plt.ylabel('Bat Landing Time to Food (seconds)')
plt.show()
