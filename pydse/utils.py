# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import, \
    unicode_literals

import sys
import logging
from itertools import chain, combinations

from six.moves import xrange
import numpy as np

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

_logger = logging.getLogger(__name__)


def atleast_2d(arr):
    """
    Ensure that an array has at least dimension 2.

    :param arr: array
    :return: array with dimension of at least 2
    """
    if arr.ndim > 1:
        return arr
    else:
        return arr[:, np.newaxis]


def powerset(iterable):
    """
    Calculates the power set of an iterable.

    :param iterable: iterable set
    :return: iterable power set
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in xrange(len(s)+1))


def make_lag_arr(lags, fuzz=1e-2):
    """
    Create a lag polynomial that can be used as 1-dim A and B for ARMA.

    This function creates a lag polynomial, i.e. 1-dim lag matrix,
    and sets to parameters that are to be estimated to the ``fuzz``
    value. Check :obj:`~.arma.ARMA.fix_constants` for further information.

    :param lags: list of lags
    :param fuzz: fill value to mark non-constant lags
    :return: tuple of the lag array and its shape
    """
    lag_size = np.max([0] + list(lags)) + 1
    pol = np.zeros(lag_size)
    pol[0], pol[list(lags)] = 1., fuzz
    shape = np.array([lag_size, 1, 1])
    return pol, shape


class UnicodeMixin(object):
    """
    Mixin class to handle defining the proper __str__/__unicode__
    methods in Python 2 or 3.
    """

    if sys.version_info[0] >= 3:  # Python 3
        def __str__(self):
            return self.__unicode__()
    else:  # Python 2
        def __str__(self):
            return self.__unicode__().encode('utf8')
