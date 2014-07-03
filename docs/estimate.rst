========================
Estimation of Parameters
========================

In this section we estimate the parameters for the time series of the monthly
passengers of an international airline.

.. plot::
    :include-source:
    :context:

    import numpy as np
    from pydse import data

    df = data.airline_passengers()
    df.plot()

.. plot::
    :include-source:
    :context:
    :nofigs:

    plt.close()
    np.random.seed(0)

Obviously, there is a strong trend in the data. Since ARMA can handle only
stationary time series we have to remove it. In order to do that, we would like
to smooth the time series. We see that there is 12 month seasonality, and
therefore taking 3 years as a window for a smoothing function should be alright.
An option would be a rolling mean:

.. plot::
    :include-source:
    :context:

    from pandas.stats.moments import rolling_mean
    df['Trend'] = rolling_mean(df['Passengers'], window=36, min_periods=1)
    df.plot()

.. plot::
    :context:

    plt.close()

Our first guess is now to remove the trend by subtracting the *Trend* from our
time series:

.. plot::
    :include-source:
    :context:

    residual = df['Passengers'] - df['Trend']
    residual.plot()

.. plot::
    :context:

    plt.close()

Obviously the trend is removed but the variance does not seem to be stationary,
i.e. there is heteroscedasticity. Since the variance seems to be related with
the absolut value of the time series we use another ansatz:

.. plot::
    :include-source:
    :context:

    residual = df['Passengers'] / df['Trend']
    residual.plot()

.. plot::
    :context:

    plt.close()

This time the series looks like a stationary process. Again, we look at the
ACF and PACF plots.

.. plot::
    :include-source:
    :context:

    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    plot_acf(residual, lags=15)

.. plot::
    :context:

    plt.close()

.. plot::
    :include-source:
    :context:

    plot_pacf(residual, lags=15)

.. plot::
    :context:

    plt.close()

These plots show us the strong seasonality of 12 months. Due to this plots, we
want to estimate an ARMA model where the *AR* term has only lag of 12 and the
*MA* has lags 1 and 13. All other lags (except of 0 of course) should be equal
to zero.

.. plot::
    :include-source:
    :context:
    :nofigs:

    from pydse.arma import ARMA

    AR = (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01]),
          np.array([13, 1, 1]))
    MA = (np.array([1, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01]),
          np.array([14, 1, 1]))
    arma = ARMA(A=AR, B=MA, rand_state=0)
    arma.fix_constants()

The `fix_constants()` functions determines the constants of our model. Every
parameter that has less or equal than one decimal place is considered constant.
Now the only remaining parameters are the ones that we set to *0.01*. In order
to estimate those we call `est_params` with our residual time series:

.. plot::
    :include-source:
    :context:

    arma.est_params(residual)

The output of this command tells us if our opimization method converged.
We can now take a look if our estimated ARMA process produces a similar time
series than residual. To quantify this similarity, we should take a look at the
Mean Absolute Deviation (MAD).

.. plot::
    :include-source:
    :context:

    import pandas as pd
    result = pd.DataFrame({'pred': arma.forecast(residual)[:, 0],
                           'truth': residual.values})
    MAD = np.mean(np.abs(result['pred'][20:] - result['truth'][20:]))
    result.plot(title="MAD: {}".format(MAD))


.. plot::
    :context:

    plt.close()

Instead of guessing the possible parameters by looking at the ACF and PACF
plots, we can also use the :obj:`~.arma.minic` function. This function takes
a set of possible AR and MA lags to consider, calculates for each combination
some information criterion and chooses the most likely.
Let's say we are quite unsure how to interpret ACF and PACF plots and we just
use our gut feeling that lag 1 and maybe lag 11, 12 as well as 13 could be
useful as AR and MA lags. We just provide those guesses to :obj:`~.arma.minic`
and get the best AR and MA lags. Then, we apply the :obj:`~.utils.make_lag_arr`
function to generate one dimensional lag matrices that we use as inputs for
our ARMA model as before. There we go:

.. plot::
    :include-source:
    :context:

    from pydse.arma import minic
    from pydse.utils import make_lag_arr
    best_ar_lags, best_ma_lags = minic([1, 11, 12, 13], [1, 11, 12, 13], residual)
    arma = ARMA(A=make_lag_arr(best_ar_lags), B=make_lag_arr(best_ma_lags), rand_state=0)
    arma.fix_constants()
    arma.est_params(residual)
    result = pd.DataFrame({'pred': arma.forecast(residual)[:, 0], 'truth': residual.values})
    MAD = np.mean(np.abs(result['pred'][20:] - result['truth'][20:]))
    result.plot(title="AR lags: {}, MA lags: {}, MAD: {}".format(best_ar_lags, best_ma_lags, MAD))

.. plot::
    :context:

    plt.close()
