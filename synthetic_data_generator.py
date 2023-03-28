import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_metric_with_daily_fall(randomize=True, linear_trend=0.0, quadratic_trend=0.0, add_anomaly=False,
                                    no_data_at_night=False):
    dates = pd.date_range(start='2023-01-01 00:00:00', end='2023-01-14 00:00:00', freq='min')

    dates_raw = []
    values_raw = []
    i = 0
    for ts in dates:
        diff = ts - ts.floor(freq='D')
        x = diff.seconds / 60 / 60
        dates_raw.append(ts)
        values_raw.append((0.1 * (x - 12) ** 3) + 200 + (i * linear_trend) + ((i * quadratic_trend) ** 2))
        i += 1

    pd_values = pd.array(values_raw)

    # Add a bit of randomness
    if randomize:
        noise = np.random.randn(len(dates_raw)) * 10 - 7
        pd_values += noise

    if add_anomaly:
        for i in range(50, 70):
            anomaly = i - 50
            pd_values[i] -= (anomaly ** 2) * 1

    if no_data_at_night:
        i = 0
        for ts in dates:
            if ts.hour < 6:
                pd_values[i] = 0
                pass
            i += 1

    pd_dates = pd.array(dates_raw)

    return pd_dates, pd_values


date_range, values = generate_metric_with_daily_fall(linear_trend=0.01, quadratic_trend=0.000001, add_anomaly=False)

frame = pd.DataFrame()
frame['timestamp'] = date_range
frame['value'] = values

frame.to_csv('data/synthetic_daily_sin_1.csv')

print(frame.info())
print(frame.head())
plt.figure(figsize=(20, 10))
plt.plot(date_range, values)
plt.show()