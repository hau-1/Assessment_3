import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np


ds1 = pd.read_csv('dataset1.csv')

ds1['start_time'] = pd.to_datetime(ds1['start_time'], dayfirst=True,)
ds1_daily = ds1.set_index('start_time').resample('D')['bat_landing_to_food'].mean().reset_index()
ds1_daily = ds1_daily.sort_values('start_time').reset_index(drop=True)

# filter
blfMean = ds1_daily['bat_landing_to_food'].mean()
blfSTD = ds1_daily['bat_landing_to_food'].std()
ds1_daily = ds1_daily[ds1_daily['bat_landing_to_food'].between(blfMean - 2*blfSTD, blfMean + 2*blfSTD)]


plt.plot(ds1_daily['start_time'], ds1_daily['bat_landing_to_food'], marker='o', color='tab:blue', label='Avg Landing Time to Food')
plt.title('Average Bat Landing Time to Food Over Time')
plt.xlabel('Date')
plt.legend()
plt.show()