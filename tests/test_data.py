# -*- coding: utf-8 -*-

"""Docstring"""

from __future__ import division, print_function, absolute_import

from pandas import DatetimeIndex

from pydse import data

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"


def test_data():
    data_funcs = [data.airline_passengers,
                  data.m1_us,
                  data.cpi_canada,
                  data.sales_product,
                  data.sales_petroleum,
                  data.sales_cola,
                  data.sales_shampoo]
    for data_func in data_funcs:
        df = data_func()
        assert isinstance(df.index, DatetimeIndex)
