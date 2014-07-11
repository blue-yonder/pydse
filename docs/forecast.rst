=====================
Forecast with horizon
=====================

In chapter :ref:`Estimation of Parameters <estimation-of-parameters>` only
one-step ahead predictions were applied. So in our example of monthly data,
all data up to last month was used to predict the number of passengers in
the current month up to the last known month.

In order to do forecasts beyond the period where all data is available, we can
provide a ``horizon`` to the :obj:`~.ARMA.forecast` function. The horizon
specifies the number of one-step ahead predictions that should be done after
the last known data point.

To illustrate this, we take the same example as
in the last chapter and remove the last two years from the data:

.. plot::
    :include-source:
    :context:

    from pydse import data
    from pydse.arma import ARMA
    from pydse.utils import make_lag_arr
    import pandas as pd
    from pandas.stats.moments import rolling_mean

    df = data.airline_passengers()
    df['Trend'] = rolling_mean(df['Passengers'], window=36, min_periods=1)
    residual_all = df['Passengers'] / df['Trend']
    residual_known = residual_all[:-24]
    pd.DataFrame({'future': residual_all, 'known': residual_known}).plot()

.. plot::
    :context:

    plt.close()

Now, we fit an ARMA model with the *known* data and forecast the following two
years after the known time period:

.. plot::
    :include-source:
    :context:

    arma = ARMA(A=make_lag_arr([1, 12, 13]), B=make_lag_arr([12]), rand_state=0)
    arma.fix_constants()
    arma.est_params(residual_known)
    pred = arma.forecast(residual_known, horizon=24)
    result = pd.DataFrame({'pred': pred[:, 0], 'truth': residual_all.values})
    result.plot()

.. plot::
    :context:

    plt.close()

By eye, it can be seen that our predictions are still quite accurate but not as
good as using one-step ahead predictions with data up to the previous month.
