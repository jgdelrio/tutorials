"""
Given m observations of humidity data for the days spanning from startDate to endDate (inclusive),
predict the hourly humidity data for each of the n timestamps in timestamps.

Complete the predictMissingHumidity function with the following 5 parameters:

1. startDate: The fist day of humidity data in the format yyyy-mm-dd
2. endDate: The last day of humidity data in the format yyyy-mm-dd
3. knownTimestamps: Each knownTimestamp(i)  (where 0 <= i <= m) denotes a yyyy-mm-dd hh:00 timestamp
                    in the inclusive range from startDate to endData that we have humidity data for
4. humidity: Each humidity(i). (where 0<= i <=m) denotes the humidity at time knownTimestamps(i)
5. timestamps: Each timestamp(i). (where 0<= j <=n) denotes a yyyy-mm-dd hh:00 timestamp in the inclusive
               range from startDate to endDate that we need to predict humidity data for

The function must return an array of n floating-point numbers where the value at each index i
detones the humidity at timestamp timestamps.
"""

from time import time
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression


TESTS = [
    (('2013-01-01', '2013-01-01',
      ['2013-01-01 07:00', '2013-01-01 08:00', '2013-01-01 09:00', '2013-01-01 10:00', '2013-01-01 11:00', '2013-01-01 12:00'],
      [10.0, 11.1, 13.2, 14.8, 15.6, 16.7],
      ['2013-01-01 13:00', '2013-01-01 14:00']),
     [18.5, 20], 5e-1),
    (('2013-01-01', '2013-01-01',
      ['2013-01-01 00:00', '2013-01-01 01:00', '2013-01-01 02:00', '2013-01-01 03:00', '2013-01-01 04:00',
       '2013-01-01 05:00', '2013-01-01 06:00', '2013-01-01 08:00', '2013-01-01 10:00',
       '2013-01-01 11:00', '2013-01-01 12:00', '2013-01-01 13:00', '2013-01-01 16:00', '2013-01-01 17:00',
       '2013-01-01 18:00', '2013-01-01 19:00', '2013-01-01 20:00', '2013-01-01 21:00', '2013-01-01 23:00'],
      [0.62, 0.64, 0.62, 0.63, 0.63, 0.64, 0.63, 0.64, 0.48, 0.46, 0.45, 0.44, 0.46, 0.47, 0.48, 0.49, 0.51, 0.52, 0.52],
      ['2013-01-01 07:00', '2013-01-01 09:00', '2013-01-01 14:00', '2013-01-01 15:00', '2013-01-01 22:00']),
     [0.57, 0.56, 0.52, 0.51, 0.45], 1e-2),
]


def predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps) -> list:
    x = [int(
        (datetime.strptime(item, "%Y-%m-%d %H:%M") - datetime.utcfromtimestamp(0)).total_seconds())
         for item in knownTimestamps]
    y = humidity

    lm = LinearRegression()
    lm.fit(np.array(x).reshape(-1, 1), y)

    new_data_points = [int(
        (datetime.strptime(item, "%Y-%m-%d %H:%M") - datetime.utcfromtimestamp(0)).total_seconds())
         for item in timestamps]
    return lm.predict(np.array(new_data_points).reshape(-1, 1))


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]
        margin = d[2]

        t0 = time()
        rst = predictMissingHumidity(*input)
        t1 = time() - t0

        diff = []
        for expected, result in zip(output, rst):
            diff.append(abs(result - expected) >= margin)

        if sum(diff) > 0:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
