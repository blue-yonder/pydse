# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import logging

import numpy as np

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

_logger = logging.getLogger(__name__)


def atleast_2d(arr):
    if arr.ndim > 1:
        return arr
    else:
        return arr[:, np.newaxis]
