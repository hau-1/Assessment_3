import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np


ds1 = pd.read_csv('dataset1.csv')

ds1['start_time'] = pd.to_datetime(ds1['start_time'], dayfirst=True,)
ds1_daily = ds1.set_index('start_time').resample('D')['seconds_after_rat_arrival'].mean().reset_index()
ds1_daily = ds1_daily.sort_values('start_time').reset_index(drop=True)

# filter
blfMean = ds1_daily['seconds_after_rat_arrival'].mean()
blfSTD = ds1_daily['seconds_after_rat_arrival'].std()
ds1_daily = ds1_daily[ds1_daily['seconds_after_rat_arrival'].between(blfMean - 2*blfSTD, blfMean + 2*blfSTD)]


plt.plot(ds1_daily['start_time'], ds1_daily['seconds_after_rat_arrival'], marker='o', color='tab:blue', label='Avg Landing Time to Food')
plt.title('Average seconds_after_rat_arrival Over Time')
plt.xlabel('Date')
plt.legend()
plt.show()