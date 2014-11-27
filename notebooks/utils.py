# -*- encoding: utf-8 -*-

"""
Helper functions used in the example notebooks.
"""

import pandas as pd
from pandas.stats.ols import OLS

__author__ = "Uwe L. Korn"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"


def ols(series):
    """
    Approximate a series by ordinary least squares regression.

    :param series: Pandas Series to be approximated
    :return: NumPy array of the same size
    """
    x = series.index.values.astype(float)
    y = series.values.astype(float)
    model = OLS(y=pd.Series(y), x=pd.Series(x))
    a = model.beta['x']
    b = model.beta['intercept']
    return a*x + b
