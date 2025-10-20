import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

ds1 = pd.read_csv('dataset1.csv')
ds1['start_time'] = pd.to_datetime(ds1['start_time'], dayfirst=True)

ds1['date'] = ds1['start_time'].dt.normalize()

def mode_or_first(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else x.iloc[0]

avg = ds1.groupby('date', as_index=False).agg({'bat_landing_to_food': 'mean','season': mode_or_first,'habit': mode_or_first})

avg['time'] = (avg['date'] - avg['date'].min()).dt.days

habit_dummies = pd.get_dummies(avg['habit'], prefix='habit', drop_first=True)
avg = pd.concat([avg, habit_dummies], axis=1)

seasons = {0: 'Winter', 1: 'Spring'}
season_metrics = {}

plt.figure(figsize=(12,6))
plt.scatter(avg['date'], avg['bat_landing_to_food'], alpha=0.4, label='Actual daily average')

for s in sorted(avg['season'].unique()):
    season_data = avg[avg['season'] == s]

    habit_column = [col for col in habit_dummies.columns]
    X_column = ['time'] + habit_column
    X_s = season_data[X_column]
    y_s = season_data['bat_landing_to_food']

    model = LinearRegression()
    model.fit(X_s, y_s)
    
    y_pred_s = model.predict(X_s)

    sort = np.argsort(season_data['time'])
    

    plt.plot(season_data['date'].iloc[sort], y_pred_s[sort],
             label=f"{seasons.get(s, f'Season {s}')} Linear Regression Line")

    mae = metrics.mean_absolute_error(y_s, y_pred_s)
    mse = metrics.mean_squared_error(y_s, y_pred_s)
    rmse = math.sqrt(mse)
    nrmse = rmse / (y_s.max() - y_s.min())
    r2 = metrics.r2_score(y_s, y_pred_s)

    print(f"\n--- {seasons.get(s, f'Season {s}')} Model ---")
    print("Intercept:", model.intercept_)
    print("Coefficients:")
    for col_name, coef in zip(X_column, model.coef_):
        print(f"  {col_name}: {coef:.4f}")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("NRMSE: ", nrmse)
    print("R^2: ", r2)

plt.xlabel('Date')
plt.ylabel('Daily Average Bat Landing to Food')
plt.title('Categorised Habit Linear Regression by Season')
plt.legend()
plt.show()
