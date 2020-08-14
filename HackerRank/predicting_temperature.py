"""
Predicting temperature

Given the hourly temperature data for each 24 hour period in p prior days spanning from startDate
to endDate (inclusive), predict the hourly temperature for the next n days starting from the day after endDate.

Complete the predictTemperature function which has 4 parameters:

1. startDate: in the format yyyy-mm-dd denoting the fist day of the p days of prior temperature data
2. endDate: in the format yyyy-mm-dd denoting the last day of the p days of prior temperature data
3. temperature: an array of 24 x p floating-point number, denoting the temperature at each timestamp
                in the inclusive range from startDate to endDate
n: integer denoting the number of future days to predict hourly temperature data for,
        starting from the day immediately following endDate

The function must return an array of 24 x n floating point numbers where each subsequest element
denotes the predicted hourly temperature data for the next n days (starting from the 1st hour of
the 1st day immediately following endDate)
"""

from time import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX


TESTS = [
    (('2013-01-01', '2013-01-01',
      [34.38, 34.36, 34.74, 35.26, 35.23, 35.29, 35.64, 36.02, 36.1, 36.98, 37.01, 36.75, 36.01, 35.66, 34.72, 33.9, 32.62, 31.51, 30.73, 29.5, 26.94, 24.56, 23.11, 21.35],
      24),
     [36.02, 36.1, 36.98, 37.01, 36.75, 36.01, 35.66, 34.72, 33.9, 32.62, 31.51, 30.73, 29.5, 26.94, 25.47, 23.84, 22.55, 21.03, 19.92, 18.77, 18.48, 18.07, 17.91, 17.11], 1),
]


def predictTemperature(startDate, endDate, temperature, n) -> list:
    p = int(len(temperature) / 24)

    x = [i for i in range(1, (24 * p) + 1)]
    y = temperature
    lm = LinearRegression()
    lm.fit(np.asarray(x).reshape(-1, 1), y)

    new_data_points = [k for k in range(x[-1] + 1, 24*n)]
    return lm.predict(np.asarray(new_data_points).reshape(-1, 1)).tolist()


def predictTemperature2(startDate, endDate, temperature, n) -> list:
    if len(temperature) != 24:
        raise ValueError('temperature must be an array with 24 elements')

    start_date = datetime.strptime(startDate, "%Y-%m-%d")
    end_date = datetime.strptime(endDate, "%Y-%m-%d") + timedelta(hours=23)

    artificial_days = 10
    artificial_start = start_date - timedelta(days=artificial_days)
    date_list = [artificial_start + timedelta(hours=x)
                 for x in range(((end_date - artificial_start).days + 1) * 24)]
    # There is too little data (1 day) and we want to predict many days so we are going to create several
    # days worth of data prior to the true data, based on the existing data with a +- randomization
    rdm_temp = np.random.choice(range(-20, 20, 1), artificial_days * 24) / 10
    full_temp = np.concatenate([rdm_temp + temperature * artificial_days, temperature])
    # Group in dataframe
    df = pd.DataFrame.from_dict({'date': date_list, 'temp': full_temp}).set_index('date')
    sarima_model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), enforce_invertibility=False,
                           enforce_stationarity=True, initialization='approximate_diffuse')
    sarima_fit = sarima_model.fit()

    new_data_points = [(date_list[-1] + timedelta(days=k)).strftime("%Y-%m-%d") for k in [1, n+1]]
    sarima_pred = sarima_fit.get_prediction(*new_data_points)
    # Filter last value of the next day
    results = sarima_pred.prediction_results._forecasts[0][:-1]

    return results


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]
        margin = d[2]

        t0 = time()
        rst = predictTemperature2(*input)
        t1 = time() - t0

        diff = []
        for expected, result in zip(output, rst):
            diff.append(abs(result - expected) >= margin)

        if sum(diff) > 0:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
