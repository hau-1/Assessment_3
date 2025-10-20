import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
 
ds2 = pd.read_csv('dataset2.csv')   

# sorting by date. bat_landing_number, food_availability, rat_arrival_number. All from ds2


ds2['time'] = pd.to_datetime(ds2['time'], dayfirst=True,)
ds2_daily = ds2.set_index('time').resample('D')['rat_arrival_number', 'food_availability', 'bat_landing_number'].mean().reset_index()
ds2_daily = ds2_daily.sort_values('time').reset_index(drop=True)

# filter

batMean = ds2_daily['bat_landing_number'].mean()
batSTD = ds2_daily['bat_landing_number'].std()
foodMean = ds2_daily['food_availability'].mean()
foodSTD = ds2_daily['food_availability'].std()
ratMean = ds2_daily['rat_arrival_number'].mean()
ratSTD = ds2_daily['rat_arrival_number'].std()
ds2_daily = ds2_daily[(ds2_daily['bat_landing_number'].between(batMean - 3*batSTD, batMean + 3*batSTD)) &
                      (ds2_daily['food_availability'].between(foodMean - 3*foodSTD, foodMean + 3*foodSTD)) &
                      (ds2_daily['rat_arrival_number'].between(ratMean - 3*ratSTD, ratMean + 3*ratSTD))]


# spearmans
spearmanBF = ds2_daily[['bat_landing_number', 'food_availability']].dropna()
spearmeanRF = ds2_daily[['rat_arrival_number', 'food_availability']].dropna()
spearmanBR = ds2_daily[['bat_landing_number', 'rat_arrival_number']].dropna()

rho_BF, pval_BF = st.spearmanr(spearmanBF['bat_landing_number'], spearmanBF['food_availability'])
rho_RF, pval_RF = st.spearmanr(spearmeanRF['rat_arrival_number'], spearmeanRF['food_availability'])
rho_BR, pval_BR = st.spearmanr(spearmanBR['bat_landing_number'], spearmanBR['rat_arrival_number'])
print(f"Spearman correlation Bat Landings vs Food Availability: rho={rho_BF:.4f}, p-value={pval_BF:.4f}")
print(f"Spearman correlation Rat Arrivals vs Food Availability: rho={rho_RF:.4f}, p-value={pval_RF:.4f}")
print(f"Spearman correlation Bat Landings vs Rat Arrivals: rho={rho_BR:.4f}, p-value={pval_BR:.4f}")

plt.plot(ds2_daily['time'], ds2_daily['rat_arrival_number'], marker='o', color='tab:red', label='Avg Rat Arrivals')
plt.plot(ds2_daily['time'], ds2_daily['food_availability'], marker='s', color='tab:green', label='Avg Food Availability')
plt.plot(ds2_daily['time'], ds2_daily['bat_landing_number'], marker='^', color='tab:blue', label='Avg Bat Landings')
plt.title('Daily Averages Over Time')
plt.xlabel('Date')
plt.legend()
plt.show()
