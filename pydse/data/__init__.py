# -*- encoding: utf-8 -*-

"""
Example data files taken from `DataMarket <http://datamarket.com/>`_.
Data is under the `default open license
<http://datamarket.com/data/license/0/default-open-license.html>`_:
"""

from __future__ import division, print_function, absolute_import

import os
import inspect
from datetime import datetime
import pandas as pd

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

__location__ = os.path.join(
    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))


def _get_df_from_file(filename):
    def parse_dates(date_str):
        for date_format in ['%y-%m', '%Y-%m']:
            try:
                return datetime.strptime(date_str, date_format)
            except ValueError:
                pass
        else:
            return ValueError("Could not parse the date {}".format(date_str))

    path = os.path.join(__location__, filename)
    return pd.read_csv(path, sep=";", parse_dates=True, index_col=0,
                       date_parser=parse_dates)


def airline_passengers():
    """
    Monthly totals of international airline passengers in thousands,
    Jan 1949 - Dec 1960
    """
    return _get_df_from_file("international-airline-passengers.csv")


def m1_us():
    """
    Monthly M1 U.S., Jan 1959 - Feb 1992
    """
    return _get_df_from_file("m1-us-1959119922.csv")


def cpi_canada():
    """
    Monthly CPI, Canada, Jan 1950 - Dec 1973
    """
    return _get_df_from_file("monthly-cpi-canada-19501973.csv")


def sales_product():
    """
    Monthly sales of a plastic manufacturer's product, Jan 2001 - May 2012
    """
    return _get_df_from_file("monthly-sales-of-product-a-for-a.csv")


def sales_cola():
    """
    Monthly sales of Tasty Cola, Jan 2001 - Mar 2012
    """
    return _get_df_from_file("monthly-sales-of-tasty-cola.csv")


def sales_shampoo():
    """
    Monthly sales of shampoo, Jan 2001 - Mar 2012
    """
    return _get_df_from_file("sales-of-shampoo-over-a-three-ye.csv")


def sales_petroleum():
    """
    Monthly sales of petroleum and related products in the U.S.,
    Jan 1971 - Dec 1991
    """
    return _get_df_from_file("us-monthly-sales-of-petroleum-an.csv")
