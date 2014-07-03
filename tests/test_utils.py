# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import logging

import numpy as np

from pydse import utils

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

logging.basicConfig(level=logging.WARN)


def test_atleast_2d():
    I = np.eye(2, 2)
    I2 = utils.atleast_2d(I)
    assert I is I2
    A = np.zeros((1, 2, 3))
    A2 = utils.atleast_2d(A)
    assert A is A2
    B = np.arange(10)
    B2 = utils.atleast_2d(B)
    assert B2.ndim == 2
    assert np.all(B2[:, 0] == B)


def test_powerset():
    result = utils.powerset([1, 2, 3])
    exp_result = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    assert list(result) == exp_result


def test_make_lag_arr():
    A, shape = utils.make_lag_arr([1, 2, 12], fuzz=999)
    assert np.all(shape == (13, 1, 1))
    assert A[0] == 1.
    assert A[1] == 999
    assert A[2] == 999
    assert A[12] == 999
    assert np.all(A[3:12] == [0]*9)
