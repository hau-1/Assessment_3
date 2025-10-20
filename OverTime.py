import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np


ds1 = pd.read_csv('dataset1.csv')
ds2 = pd.read_csv('dataset2.csv')


ds2['time'] = pd.to_datetime(ds2['time'], dayfirst=True)
cutoff_date = pd.to_datetime('18/03/2018', dayfirst=True)
#daily_avg_food = ds2.groupby('time')['food_availability'].mean().reset_index()
daily_avg_bats = ds2.groupby('time')['bat_landing_number'].mean().reset_index()


plt.plot(daily_avg_bats['time'], daily_avg_bats['bat_landing_number'])
plt.xlabel('Date')
plt.ylabel('Average Bat La')
plt.title('Daily Average Bat Arrivals Over Time')
plt.show()