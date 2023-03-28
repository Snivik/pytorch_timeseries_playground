import pandas as pd
import random

DAILY_MATRIX_OF_SALES = [
    10.0, # 00:00
    10,
    10,
    10,
    10,
    10,
    10, # 06:00
    15,
    25,
    30,
    25,
    30,
    45, # 12:00
    50,
    45,
    40,
    45,
    70,
    80, # 18:00
    90,
    100,
    60, # 21:00
    30,
    10,
]

def generate_matrix_based_data(anomalous_day=-1, scale_factor=2, rand_fact=0.2):
    dates = pd.date_range(start='2023-01-01 00:00:00', end='2023-01-22 00:00:00', freq='h')
    values_raw = []

    for ts in dates:

        value = DAILY_MATRIX_OF_SALES[ts.hour]
        rand = value * (random.random() * rand_fact) # Add 0-20%
        rand -= value * (rand_fact/2) # Sub 10% so that diff is +/-10%

        value += rand
        # Every day at 11 pm we report value
        if ts.day == anomalous_day:
            value *= scale_factor

        values_raw.append(value)


    return dates, pd.array(values_raw)
