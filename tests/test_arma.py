# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import logging

import numpy as np
import numpy.testing as nptest
import pytest

from pydse.arma import ARMA, ARMAError

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

logging.basicConfig(level=logging.WARN)


def test_arma_construction():
    AR = ([1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3], [3, 2, 2])
    MA = ([1, .2, 0, .1, 0, 0, 1, .3], [2, 2, 2])
    X = ([1, 2, 3, 4, 5, 6], [1, 2, 3])
    # Check construction
    ARMA(A=AR, B=MA)
    ARMA(A=AR)
    ARMA(A=AR, B=MA, C=X)

    MA = ([1, 0.2, 0, .1], [2, 2, 1])
    X = ([1, 2, 3, 4, 5, 6], [2, 1, 3])
    with pytest.raises(ARMAError):
        ARMA(A=AR, B=MA)
        ARMA(A=AR, C=X)

    AR = ([1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3], [3, 2, 2])
    MA = ([1, 2, 0, .1, 0, 0, 1, .3], [2, 2, 2])
    TREND = [1, 2]
    ARMA(A=AR, B=MA, TREND=TREND)
    TREND = [[1, 2], [3, 4]]
    ARMA(A=AR, B=MA, TREND=TREND)
    TREND = [[1, 2, 3], [3, 4, 5]]
    ARMA(A=AR, B=MA, TREND=TREND)
    TREND = [1, 2, 3]
    # give a (3,) array while expect a (2,) array as p = 2
    with pytest.raises(ARMAError):
        ARMA(A=AR, B=MA, TREND=TREND)
    TREND = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    # give a (3, 3) array while expect a (2, X) array as p = 2
    with pytest.raises(ARMAError):
        ARMA(A=AR, B=MA, TREND=TREND)
    TREND = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    # give a (3, 3, 3) array while expect a 2-d matrix
    with pytest.raises(ARMAError):
        ARMA(A=AR, B=MA, TREND=TREND)


def test_simulate():
    AR = (np.array([1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3]),
          np.array([3, 2, 2]))
    MA = (np.array([1, .2, 0, .1, 0, 0, 1, .3]), np.array([2, 2, 2]))
    arma = ARMA(A=AR, B=MA, C=None)
    # noise generated with R for comparison
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
    # R simulation results ( arma <- ARMA(A=AR, B=MA, C=NULL);
    # R_result <- simulate(arma, noise=noise, sampleT=10) )
    series0 = np.array(
        [1.58239012, 0.85063747, -0.11981462, -1.69017627, -0.19912156,
         0.02830831, 0.16284912, 2.42364792, -0.15007052, -1.27531927])
    series1 = np.array(
        [-0.5172168, -0.4261651, 0.2958942, -0.8559883, 0.7033546, -1.0857290,
         -0.2788928, 0.7393030, -0.2999778, 0.6363970])
    R_result = np.vstack([series0, series1]).T

    result = arma.simulate(sampleT=10, noise=noise)
    nptest.assert_almost_equal(result, R_result)

    # R simulations results with trend ( TREND <- c(1,2);
    # arma_trend <- ARMA(A=AR, B=MA, C=NULL, TREND=TREND);
    # R_result_trend <- simulate(arma_trend, noise=noise, sampleT=sampleT) )
    TREND = np.array([1., 2.])
    arma = ARMA(A=AR, B=MA, C=None, TREND=TREND)
    series0 = np.array(
        [2.5823901, 0.9506375, 0.2701854, -1.1311763, 0.1139784, 0.4486183,
         0.6048481, 2.8091458, 0.2663152, -0.8590244])
    series1 = np.array(
        [1.48278321, 0.37383493, 1.17589420, 0.37601165, 1.67255459,
         -0.05844899, 0.80133515, 1.76057423, 0.74401876, 1.68619050])
    R_result = np.vstack([series0, series1]).T
    result = arma.simulate(sampleT=10, noise=noise)
    nptest.assert_almost_equal(result, R_result)


def test_non_consts():
    AR = (np.array([1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3]),
          np.array([3, 2, 2]))
    MA = (np.array([1, .2, 0, .1, 0, 0, 1, .3]),
          np.array([2, 2, 2]))
    arma = ARMA(A=AR, B=MA, C=None)
    arma.Aconst[:, 0, 0] = True
    arma.Bconst[0, :, 0] = True
    new_values = np.repeat(-999., 9 + 6)
    arma.non_consts = new_values
    result_A = np.array([1, .5, .3, -999., -999., -999., -999., -999., -999.,
                         -999., -999., -999.]).reshape((3, 2, 2), order='F')
    nptest.assert_array_equal(arma.A, result_A)
    result_B = np.array([1, -999., 0, -999., -999., -999., -999.,
                         -999.]).reshape((2, 2, 2), order='F')
    nptest.assert_array_equal(arma.B, result_B)
    set_values = arma.non_consts
    nptest.assert_array_equal(set_values, new_values)


def test_forecast():
    AR = (np.array([1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3]),
          np.array([3, 2, 2]))
    MA = (np.array([1, .2, 0, .1, 0, 0, 1, .3]),
          np.array([2, 2, 2]))
    arma = ARMA(A=AR, B=MA, C=None)

    series0 = np.array(
        [1.58239012, 0.85063747, -0.11981462, -1.69017627, -0.19912156,
         0.02830831, 0.16284912, 2.42364792, -0.15007052, -1.27531927])
    series1 = np.array(
        [-0.5172168, -0.4261651, 0.2958942, -0.8559883, 0.7033546, -1.0857290,
         -0.2788928, 0.7393030, -0.2999778, 0.6363970])
    R_result = np.vstack([series0, series1]).T

    # One ahead forecast generated with R
    # ( arma <- l(arma, R_result); R_pred <- arma$estimates$pred )
    pred0 = np.array(
        [0.00000000, -0.37127368, -0.54455969, -0.14820550, 0.72904133,
         0.32310958, 0.16860013, 0.01899776, -0.91366463, -0.47630989])
    pred1 = np.array(
        [0.00000000, -0.05479565, 0.05066136, 0.03484596, 0.26779526,
         0.15181266, -0.05463401, 0.36191171, -0.43331575, -0.16779191])
    R_pred = np.vstack([pred0, pred1]).T

    pred = arma.forecast(y=R_result)
    nptest.assert_almost_equal(R_pred, pred)

    # One ahead forecast with trend generated with R
    # ( arma_trend <- l(arma_trend, R_result_trend);
    # R_pred_trend <- arma_trend$estimates$pred )
    TREND = np.array([1., 2.])
    arma_trend = ARMA(A=AR, B=MA, C=None, TREND=TREND)
    series0 = np.array(
        [2.5823901, 0.9506375, 0.2701854, -1.1311763, 0.1139784, 0.4486183,
         0.6048481, 2.8091458, 0.2663152, -0.8590244])
    series1 = np.array(
        [1.48278321, 0.37383493, 1.17589420, 0.37601165, 1.67255459,
         -0.05844899, 0.80133515, 1.76057423, 0.74401876, 1.68619050])
    R_result_trend = np.vstack([series0, series1]).T

    pred0 = np.array(
        [1.00000000, -0.27127368, -0.15455969, 0.41079450, 1.04214133,
         0.74341958, 0.61059913, 0.40449566, -0.49727892, -0.06001498])
    pred1 = np.array(
        [2.00000000, 0.7452043, 0.9306614, 1.2668460, 1.2369953, 1.1790927,
         1.0255940, 1.3831829, 0.6106808, 0.8820015])
    R_pred_trend = np.vstack([pred0, pred1]).T

    pred_trend = arma_trend.forecast(y=R_result_trend)
    nptest.assert_almost_equal(R_pred_trend, pred_trend)


def test_fix_constants():
    AR = (np.array([1, .5, .31, 0, .2, .1, 0, .2, .01, 1, .49, .3]),
          np.array([3, 2, 2]))
    MA = (np.array([1, .21, 0, .1, 0, -0.01, 1, .3]), np.array([2, 2, 2]))
    arma = ARMA(A=AR, B=MA, C=None)
    arma.fix_constants()
    Aconst = np.array(
        [[[True, True], [True, True]], [[True, True], [True, False]],
         [[False, False], [True, True]]], dtype=bool)
    nptest.assert_array_equal(arma.Aconst, Aconst)
    Bconst = np.array(
        [[[True, True], [True, True]], [[False, False], [True, True]]],
        dtype=bool)
    nptest.assert_array_equal(arma.Bconst, Bconst)


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

    negloglike = arma.negloglike(pred, result)
    R_negloglike = 25.4247320523
    nptest.assert_almost_equal(negloglike, R_negloglike)


def test_est_params():
    rand_state = np.random.RandomState()
    rand_state.seed(42)

    # Generate a target series
    AR = (np.array([1, 0.3, 0.5]), np.array([3, 1, 1]))
    MA = (np.array([1, .1]), np.array([2, 1, 1]))
    arma = ARMA(A=AR, B=MA, C=None, rand_state=rand_state)
    series = arma.simulate(sampleT=1000)

    # Estimate parameters by simulated series
    AR_est = (np.array([1, .01, .01]), np.array([3, 1, 1]))
    MA_est = (np.array([1, .01]), np.array([2, 1, 1]))
    arma_est = ARMA(A=AR_est, B=MA_est, C=None)
    arma_est.fix_constants()
    arma_est.est_params(y=series)

    nptest.assert_almost_equal(arma_est.A, arma.A, decimal=1)
    nptest.assert_almost_equal(arma_est.B, arma.B, decimal=1)
