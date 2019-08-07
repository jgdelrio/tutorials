import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries, window=12, graph=True, verbose=True):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    if graph:
        # Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        fig = plt.gcf()
        fig.set_size_inches(14, 6)
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

    # Perform Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    if verbose:
        print('Results of Dickey-Fuller Test:')
        print(dfoutput)

    return dfoutput
