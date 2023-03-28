import pandas as pd
import random

def generate_metric_with_one_daily_report():
    dates = pd.date_range(start='2023-01-01 00:00:00', end='2023-01-15 00:00:00', freq='h')
    values_raw = []

    for ts in dates:

        # Every day at 11 pm we report value
        if ts.hour == 23:
            values_raw.append(random.random() * 5 + 90)
        else:
            values_raw.append(1)

    return dates, pd.array(values_raw)