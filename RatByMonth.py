import pandas as pd
import matplotlib.pyplot as plt
import calendar

ds2 = pd.read_csv('dataset2.csv')

ds2['time'] = pd.to_datetime(ds2['time'], dayfirst=True,)
ds2['rat_arrival_number'] = pd.to_numeric(ds2['rat_arrival_number'])
ds2 = ds2.dropna(subset=['time','rat_arrival_number']).copy()


weekly = ds2.set_index('time').resample('W-MON')['rat_arrival_number'].mean().reset_index()
weekly = weekly.dropna().reset_index(drop=True)

week_labels = weekly['time'].dt.strftime('%Y-%m-%d')

plt.figure(figsize=(12,6))
plt.bar(week_labels, weekly['rat_arrival_number'], color='tab:green', edgecolor='k')
plt.xticks(rotation=45)

plt.xlabel('Week start (YYYY-MM-DD)')
plt.ylabel('Average rat_arrival_number')
plt.title('Average rat_arrival_number by Week')
plt.tight_layout()
plt.show()