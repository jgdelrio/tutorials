import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson


def adjust(val, length=6):
    return str(val).ljust(length)


def test_stationarity(ts, window=12, autolag='AIC', signif=0.05,
                      name='', graph=True, verbose=True):
    """
    Verify the stationarity of a Time Series
    :param ts: Time Series
    :param window:     window required (ex: 12 months)
    :param autolag:    parameter of the Dickey-Fuller test (adfuller function)
    :param name:       name of the series (optional)
    :param graph:      (boolean) plot the resulting graph
    :param verbose:    (boolean) verbose mode
    :return:           dataframe with the information generated (statistics, p-value, #lags, etc)
    """
    # Determing rolling statistics
    rolmean = ts.rolling(window).mean()
    rolstd = ts.rolling(window).std()

    if graph:
        # Plot rolling statistics:
        orig = plt.plot(ts, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        fig = plt.gcf()
        fig.set_size_inches(14, 6)
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

    # Perform Dickey-Fuller test:
    dftest = adfuller(ts, autolag=autolag)
    output = {'test_statistic': round(dftest[0], 4),
              'pvalue': round(dftest[1], 4),
              'n_lags': round(dftest[2], 4),
              'n_obs': dftest[3]}
    p_value = output['pvalue']

    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    if verbose:
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n", '-' * 47)
        print(dfoutput)

        print(f"Significance Level: {signif}")
        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")

    return dfoutput


def approximate_entropy(U, m, r):
    """Compute aproximate-entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m+1) - _phi(m))


def sample_entropy(U, m, r):
    """Compute Sample entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m+1) / _phi(m))


def grangers_causation_matrix(data, variables, test='ssr_chi2test', maxlag=12, verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table are the P-Values.
    P-Values lesser than the significance level 0.05, implies that we reject the Null Hypothesis that
    there is not causality (or that the coefficients of the corresponding past values are zero), so we
    conclude that there is causality.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def cointegration_test(df, alpha=0.05, verbose=True):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]

    dfoutput = pd.DataFrame(zip(df.columns, traces, cvts, traces>cvts),
                            columns=['Name', 'Test Stat', '> C(95%)', 'Significance']).set_index("Name")
    if verbose:
        # Summary
        print(dfoutput)

    return dfoutput


def durbin_watson_statistic(model, columns=None, verbose=True):
    """
    This is  used to check if there is any leftover pattern in the residuals (errors).
    Check for Serial Correlation of Residuals (Errors) using Durbin Watson Statistic (DW).

    Basically if there is any correlation left in the residuals, then there is some pattern in the
    time series that is still left to be explained by the model. In that case, the typical course
    of action is to either increase the order of the model or induce more predictors into the system
    or look for a different algorithm to model the time series.

    DW can vary from 0 to 4:
    - close to 2: then there is no significant serial correlation
    - close to 0: positive serial correlation
    - close to 4: negative serial correlation

    :param model: model already fitted
    :return:

    """
    out = durbin_watson(model.resid)

    if verbose:
        if columns is None:
            print(f"Please specify the list of columns in which the model was fitted")
        else:
            if not isinstance(columns, (list, pd.core.indexes.base.Index)):
                raise TypeError(f"columns must be a list")
            for col, val in zip(columns, out):
                print(adjust(col), ':', round(val, 3))

    return out


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns

    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc
