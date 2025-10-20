import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
# ...existing code...

# load dataset1 and prepare datetime
ds1 = pd.read_csv('dataset1.csv')
if 'start_time' not in ds1.columns or 'bat_landing_to_food' not in ds1.columns:
    raise KeyError("dataset1.csv must contain 'start_time' and 'bat_landing_to_food' columns")

ds1['start_time'] = pd.to_datetime(ds1['start_time'], dayfirst=True, errors='coerce')
ds1 = ds1.dropna(subset=['start_time','bat_landing_to_food']).copy()

# --- remove outliers from ds1['bat_landing_to_food'] (IQR method) ---
method = 'iqr'   # options: 'iqr', 'zscore', 'mad'
s = ds1['bat_landing_to_food']

if method == 'zscore':
    mask = (s - s.mean()).abs() <= 3 * s.std()
elif method == 'iqr':
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = s.between(lower, upper)
elif method == 'mad':
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad == 0:
        mask = pd.Series(True, index=s.index)
    else:
        mod_z = 0.6745 * (s - med) / mad
        mask = mod_z.abs() <= 3.5
else:
    raise ValueError("Unknown outlier filter method")

removed = len(ds1) - int(mask.sum())
ds1 = ds1[mask].copy()
print(f"ds1 outlier filter='{method}': removed {removed} rows, remaining {len(ds1)}")

# group by calendar month (1..12) and compute monthly means
ds1['month'] = ds1['start_time'].dt.month
monthly = ds1.groupby('month')['bat_landing_to_food'].mean().reindex(range(1,13))

# month labels
import calendar
months = [calendar.month_name[m] for m in monthly.index]

# plot bar chart
plt.figure(figsize=(10,5))
plt.bar(months, monthly, color='tab:blue', edgecolor='k')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Average bat_landing_to_food')
plt.title('Average bat_landing_to_food by Month')
plt.tight_layout()
plt.show()

# convert to monthly time series (year-month) and compute monthly mean
ds1['start_time'] = pd.to_datetime(ds1['start_time'], dayfirst=True, errors='coerce')
monthly_ts = ds1.set_index('start_time').resample('M')['bat_landing_to_food'].mean()

# create full monthly index from first to last month and reindex (introduces NaNs for gaps)
full_idx = pd.date_range(start=monthly_ts.index.min(), end=monthly_ts.index.max(), freq='M')
monthly_ts = monthly_ts.reindex(full_idx)

# numeric x = days since start (works for regression)
start_date = monthly_ts.index.min()
x_all = (monthly_ts.index - start_date).days.astype(float)

# observed (non-missing) points for fitting
obs = monthly_ts.dropna()
x_obs = (obs.index - start_date).days.astype(float)
y_obs = obs.values

# choose regression degree (1 = linear, 2 = quadratic)
deg = 1
if len(x_obs) < deg + 2:
    raise RuntimeError("Not enough observed months to fit requested polynomial degree")

# fit polynomial with numpy
coef = np.polyfit(x_obs, y_obs, deg)
y_fit_all = np.polyval(coef, x_all)

# fill only the missing months with predictions (keep observed values)
monthly_filled = monthly_ts.copy()
missing_mask = monthly_filled.isna()
monthly_filled.loc[missing_mask] = y_fit_all[missing_mask.values]

# plot original, filled and regression
plt.figure(figsize=(10,5))
plt.plot(monthly_ts.index, y_fit_all, color='gray', linestyle='--', label=f'Poly deg={deg} fit')
plt.xlim(monthly_ts.index.min(), monthly_ts.index.max())
plt.scatter(monthly_ts.index, monthly_ts.values, color='tab:blue', label='Observed (monthly mean)')
plt.scatter(monthly_filled.index[missing_mask], monthly_filled[missing_mask], 
            color='tab:orange', marker='s', label='Filled (predicted)')
plt.xlabel('Month')
plt.ylabel('bat_landing_to_food')
plt.title('Monthly mean bat_landing_to_food — regression fill for gaps')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# optional: replace monthly values in ds1-derived structure with filled series for downstream use
# monthly_filled_series = monthly_filled  # use this for further analysis

# group by calendar week (weeks starting Monday) and compute weekly means
weekly_ts = ds1.set_index('start_time').resample('W-MON')['bat_landing_to_food'].mean()

# create labels (use week end date) and plot bar chart
week_labels = weekly_ts.index.strftime('%Y-%m-%d')  # strings for x-axis
plt.figure(figsize=(12,6))
plt.bar(week_labels, weekly_ts.values, color='tab:blue', edgecolor='k')
plt.xticks(rotation=45)

# reduce x-tick label density if many weeks
if len(week_labels) > 20:
    step = max(1, len(week_labels)//20)
    for i, lbl in enumerate(plt.gca().xaxis.get_ticklabels()):
        lbl.set_visible(i % step == 0)

plt.xlabel('Week (YYYY-MM-DD)')
plt.ylabel('Average bat_landing_to_food')
plt.title('Average bat_landing_to_food by Week')
plt.tight_layout()
plt.show()

# group by calendar day and compute daily means
daily_ts = ds1.set_index('start_time').resample('D')['bat_landing_to_food'].mean()

# create labels (use day) and plot bar chart
day_labels = daily_ts.index.strftime('%Y-%m-%d')
plt.figure(figsize=(14,6))
plt.bar(day_labels, daily_ts.values, color='tab:blue', edgecolor='k')
plt.xticks(rotation=45)

# reduce x-tick label density if many days
if len(day_labels) > 30:
    step = max(1, len(day_labels)//30)
    for i, lbl in enumerate(plt.gca().xaxis.get_ticklabels()):
        lbl.set_visible(i % step == 0)

plt.xlabel('Date (YYYY-MM-DD)')
plt.ylabel('Average bat_landing_to_food (daily)')
plt.title('Average bat_landing_to_food by Day')
plt.tight_layout()
plt.show()