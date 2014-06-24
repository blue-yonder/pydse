# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import logging

import numpy as np
import numpy.testing as nptest
import pytest

from pydse.arma import ARMA, ARMAError
from pydse import stats

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

logging.basicConfig(level=logging.WARN)


def test_negloglike():
    AR = (np.array([1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3]),
          np.array([3, 2, 2]))
    MA = (np.array([1, .2, 0, .1, 0, 0, 1, .3]),
          np.array([2, 2, 2]))
    arma = ARMA(A=AR, B=MA, C=None)
    # Noise generated with R for comparison
    # ( setRNG(seed=0); noise <- makeTSnoise(10, 2, 2) )
    w0_series0 = np.array([1.2629543, -0.3262334])
    w0_series1 = np.array([-1.1476570, -0.2894616])
    w_series0 = np.array(
        [1.329799263, 1.272429321, 0.414641434, -1.539950042, -0.928567035,
         -0.294720447, -0.005767173, 2.404653389, 0.763593461, -0.799009249])
    w_series1 = np.array(
        [-0.2992151, -0.4115108, 0.2522234, -0.8919211, 0.4356833, -1.2375384,
         -0.2242679, 0.3773956, 0.1333364, 0.8041895])
    noise = (np.vstack([w0_series0, w0_series1]).T,
             np.vstack([w_series0, w_series1]).T)

    result = arma.simulate(sampleT=10, noise=noise)
    pred = arma.forecast(y=result)

    negloglike = stats.negloglike(pred, result)
    R_negloglike = 25.4247320523
    nptest.assert_almost_equal(negloglike, R_negloglike)