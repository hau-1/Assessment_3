import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics


ds1 = pd.read_csv('dataset1.csv')
ds1['start_time'] = pd.to_datetime(ds1['start_time'], dayfirst=True)

ds1['date'] = ds1['start_time'].dt.normalize() 

avg = ds1.groupby('date')['bat_landing_to_food', 'season'].mean().reset_index()

avg['time'] = (avg['date'] - avg['date'].min()).dt.days

X = avg[['time', 'season']]
y = avg['bat_landing_to_food']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
avg['predicted'] = y_pred

mae = metrics.mean_absolute_error(y, y_pred)
mse = metrics.mean_squared_error(y, y_pred)
rmse = math.sqrt(metrics.mean_squared_error(y, y_pred))
nrmse = rmse / (y.max() - y.min())
r2 = metrics.r2_score(y, y_pred)

print("MAE:", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("NRMSE: ", nrmse)
print("R^2: ", r2)

seasons = {0: 'Winter', 1: 'Spring'}

plt.figure(figsize=(12,6))
plt.scatter(avg['date'], avg['bat_landing_to_food'], label='Average Bat Landing Time to Food', alpha=0.6)

season_metrics = {}
for s in sorted(avg['season'].unique()):
    season_data = avg[avg['season'] == s]
    X_s = season_data[['time']]
    y_s = season_data['bat_landing_to_food']

    model = LinearRegression()
    model.fit(X_s, y_s)
    y_pred_s = model.predict(X_s)

    sort = np.argsort(season_data['time'])
    plt.plot(season_data['date'].iloc[sort], y_pred_s[sort],
             label=f'{seasons.get(s, f"Season {s}")} Linear Regression Line')

    mae_s = metrics.mean_absolute_error(y_s, y_pred_s)
    mse_s = metrics.mean_squared_error(y_s, y_pred_s)
    rmse_s = math.sqrt(mse_s)
    nrmse_s = rmse_s / (y_s.max() - y_s.min())
    r2_s = metrics.r2_score(y_s, y_pred_s)

    print(f" {seasons.get(s,'')} Performance Metrics")
    print("Intercept:", model.intercept_)
    print("Coefficient (time):", float(model.coef_[0]))
    print("MAE: ", mae_s)
    print("MSE: ", mse_s)
    print("RMSE: ", rmse_s)
    print("NRMSE: ", nrmse_s)
    print("R^2: ", r2_s)


plt.xlabel('Date')
plt.ylabel('Daily Average Bat Landing to Food')
plt.title('Daily Average Bat Landing to Food Over Time with Linear Regression by Season')
plt.legend()
plt.show()
