# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import logging
import operator
import itertools

import numpy as np
from numpy import linalg
from scipy import optimize
from six.moves import xrange

from . import utils
from . import stats

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

_logger = logging.getLogger(__name__)


class ARMAError(Exception):
    def __unicode__(self):
        return unicode(self.message)

    def __str__(self):
        return unicode(self).encode('utf-8')


class ARMA(object):
    """
    A(L)y(t) = B(L)e(t) + C(L)u(t) - TREND(t)

    * L: Lag/Shift operator,
    * A: (axpxp) tensor to define auto-regression,
    * B: (bxpxp) tensor to define moving-average,
    * C: (cxpxm) tensor for external input,
    * e: (pxt) matrix of unobserved disturbance (white noise),
    * y: (pxt) matrix of observed output variables,
    * u: (mxt) matrix of input variables,
    * TREND: (pxt) matrix like y or a p-dim vector.

    If B is net set, fall back to VAR, i.e. B(L) = I.
    """
    def __init__(self, A, B=None, C=None, TREND=None, rand_state=None):
        self.A = np.asarray(A[0]).reshape(A[1], order='F')
        if B is not None:
            self.B = np.asarray(B[0]).reshape(B[1], order='F')
        else:
            # Set B(L) = I
            shape = A[1][1:]
            self.B = np.empty(shape=np.hstack(([1], shape)))
            self.B[0] = np.eye(*shape)
        if C is not None:
            self.C = np.asarray(C[0]).reshape(C[1], order='F')
        else:
            self.C = np.empty((0, 0, 0))
        if TREND is not None:
            self.TREND = np.asarray(TREND)
        else:
            self.TREND = None
        self._check_consistency(self.A, self.B, self.C, self.TREND)

        self.Aconst = np.zeros(self.A.shape, dtype=np.bool)
        self.Bconst = np.zeros(self.B.shape, dtype=np.bool)
        self.Cconst = np.zeros(self.C.shape, dtype=np.bool)

        if rand_state is None:
            self.rand = np.random.RandomState()
        elif isinstance(rand_state, np.random.RandomState):
            self.rand = rand_state
        else:
            self.rand = np.random.RandomState(rand_state)

    def _get_num_non_consts(self):
        a = np.sum(~self.Aconst)
        b = np.sum(~self.Bconst)
        c = np.sum(~self.Cconst)
        return a, b, c

    @property
    def non_consts(self):
        a = self.A[~self.Aconst]
        b = self.B[~self.Bconst]
        c = self.C[~self.Cconst]
        return np.hstack([a, b, c])

    @non_consts.setter
    def non_consts(self, values):
        parts = np.cumsum(self._get_num_non_consts())
        if values.size != parts[2]:
            raise ARMAError("Number of values does not equal number "
                            "of non-constants")
        self.A[~self.Aconst] = values[:parts[0]]
        self.B[~self.Bconst] = values[parts[0]:parts[1]]
        self.C[~self.Cconst] = values[parts[1]:parts[2]]

    def _check_consistency(self, A, B, C, TREND):
        if A is None:
            raise ARMAError("A needs to be set for an ARMA model")
        n = A.shape[1]
        if n != A.shape[2] or len(A.shape) > 3:
            raise ARMAError("A needs to be of shape (a, p, p)")
        if n != B.shape[1] or (n != B.shape[2] or len(B.shape) > 3):
            raise ARMAError("B needs to be of shape (b, p, p) with A being "
                            "of shape (a, p, p)")
        if C.size != 0 and (n != C.shape[1] or len(C.shape) > 3):
            raise ARMAError("C needs to be of shape (c, p, m) with A being "
                            "of shape (a, p, p)")
        if TREND is not None:
            if len(TREND.shape) > 2:
                raise ARMAError("TREND needs to of shape (p, t) with A being "
                                "of shape (a, p, p)")
            elif len(TREND.shape) == 2 and n != TREND.shape[0]:
                raise ARMAError("TREND needs to of shape (p, t) with A being "
                                "of shape (a, p, p)")
            elif len(TREND.shape) == 1 and n != TREND.shape[0]:
                raise ARMAError("TREND needs to of shape (p, t) with A being "
                                "of shape (a, p, p)")

    def _get_noise(self, samples, p, lags):
        w0 = self.rand.normal(size=lags * p).reshape((lags, p))
        w = self.rand.normal(size=samples * p).reshape((samples, p))
        return w0, w

    def _prep_trend(self, trend, dim_t, dim_p):
        if trend is not None:
            if len(trend.shape) == 2:
                assert trend.shape[1] == dim_t
                return np.copy(trend)
            else:
                return np.tile(trend, (dim_t, 1))
        else:
            return np.zeros((dim_t, dim_p))

    def simulate(self, y0=None, u0=None, u=None, sampleT=100, noise=None):
        p = self.A.shape[1]
        a, b = self.A.shape[0], self.B.shape[0]
        c, m = self.C.shape[0], self.C.shape[2]
        y0 = utils.atleast_2d(y0) if y0 is not None else np.zeros((a, p))
        u = utils.atleast_2d(u) if u0 is not None else np.zeros((c, m))
        u0 = utils.atleast_2d(u0) if u0 is not None else np.zeros((c, m))
        if noise is None:
            noise = self._get_noise(sampleT, p, b)
        w0, w = noise
        assert y0.shape[0] >= a
        assert w0.shape[0] >= b
        assert u0.shape[0] >= c

        # diagonalize with respect to matrix of A's leading coefficients
        A0inv = linalg.inv(self.A[0, :, :])
        A = np.tensordot(self.A, A0inv, axes=1)
        B = np.tensordot(self.B, A0inv, axes=1)
        if c != 0:
            C = np.einsum('ijk,kl', self.C, A0inv)
        else:
            C = np.zeros((c, p, m))

        # prepend start values to the series
        y = self._prep_trend(self.TREND, sampleT, p)
        y = np.vstack((y0[a::-1, ...], y))
        w = np.vstack((w0[b::-1, ...], w))
        u = np.vstack((u0[c::-1, ...], u))
        
        # perform simulation by multiplying the lagged matrices to the vectors
        # and summing over the different lags
        for t in xrange(a, sampleT+a):
            y[t, :] -= np.einsum('ikj, ij', A[1:, ...], y[t-1:t-a:-1, :])
            if b != 0:
                y[t, :] += np.einsum('ikj, ij', B, w[t-a+b:t-a:-1, :])
            if c != 0:
                y[t, :] += np.einsum('ikj, ij', C, u[t-a+b:t-a:-1, :])
        return y[a:]

    def forecast(self, y, horizon=0, u=None):
        p = self.A.shape[1]
        a, b = self.A.shape[0], self.B.shape[0]
        c, m = self.C.shape[0], self.C.shape[2]
        u = u if u is not None else np.zeros((c, m))
        if y.ndim == 1:
            y = utils.atleast_2d(y)

        sampleT = y.shape[0]
        predictT = sampleT + horizon

        # diagonalize with respect to matrix of B's leading coefficients
        B0inv = linalg.inv(self.B[0, :, :])
        A = np.tensordot(self.A, B0inv, axes=1)
        B = np.tensordot(self.B, B0inv, axes=1)
        if c != 0:
            C = np.einsum('ijk,kl', self.C, B0inv)
        else:
            C = np.zeros((c, p, m))

        # calculate directly the residual ...
        res = -np.dot(self._prep_trend(self.TREND, sampleT, p), B0inv)
        # and perform prediction
        for t in xrange(sampleT):
            la, lb, lc = min(a-1, t), min(b-1, t), min(c-1, t)
            ba, bb, bc = max(0, t-la), max(0, t-lb), max(0, t-lc)
            res[t, :] += np.einsum('ikj,ij', A[la::-1, ...], y[ba:t+1, :])
            if b != 0:
                res[t, :] -= np.einsum('ikj,ij', B[lb:0:-1, ...], res[bb:t, :])
            if c != 0:
                res[t, :] -= np.einsum('ikj,ij', C[lc::-1, ...], u[bc:t+1, :])

        pred = np.zeros((predictT, p))
        pred[:sampleT, :] = y[:sampleT, :] - np.dot(res, B[0, :, :])

        # ToDo: Implement this!
        if predictT > sampleT:
            pass

        return pred

    def fix_constants(self, fuzz=1e-5, prec=1):
        @np.vectorize
        def is_const(x):
            return abs(x - round(x, prec)) < fuzz

        def set_const(M, Mconst):
            M_mask = is_const(M)
            Mconst[M_mask] = True
            Mconst[~M_mask] = False

        set_const(self.A, self.Aconst)
        set_const(self.B, self.Bconst)
        if self.C.size != 0:
            set_const(self.C, self.Cconst)

    def est_params(self, y):
        y = utils.atleast_2d(y)

        def cost_function(x):
            self.non_consts = x
            pred = self.forecast(y=y)
            return stats.negloglike(pred, y)

        x0 = self.non_consts
        return optimize.minimize(cost_function, x0)


def minic(ar_lags, ma_lags, y, crit='BIC'):
    """
    Minimum information citerion method to fit ARMA.

    Use the Akaike information criterion (AIC) or
    Bayesian information criterion (BIC) to determine the
    most promising AR and MA lags for an ARMA model.

    This method only works for scalar time series, i.e.
    dim(y[0]) = 1.

    :param ar_lags: list of AR lags to consider
    :param ma_lags: list of MA lags to consider
    :param y: target vector or scalar time series
    :param crit: information criterion ('BIC' or 'AIC')
    :return: tuple of AR lags and MA lags
    """
    assert y.ndim == 1
    all_ar_lags = list(utils.powerset(sorted(ar_lags)))
    all_ma_lags = list(utils.powerset(sorted(ma_lags)))
    lags = itertools.product(all_ar_lags, all_ma_lags)
    lags.next()  # drop case with no AR and MA lags
    metric = dict()  # metric
    for ar_lags, ma_lags in lags:
        arma = ARMA(A=utils.make_lag_arr(ar_lags),
                    B=utils.make_lag_arr(ma_lags))
        arma.fix_constants()
        ret_val = arma.est_params(y)
        if ret_val['success']:
            k = len(ar_lags) + len(ma_lags)
            nloglike = ret_val['fun']
            if crit == 'BIC':
                metric[(ar_lags, ma_lags)] = stats.bic(nloglike, k, len(y))
            elif crit == 'AIC':
                metric[(ar_lags, ma_lags)] = stats.aic(nloglike, k)
            else:
                raise RuntimeError("Unknown method")
        else:
            metric[(ar_lags, ma_lags)] = np.inf
    return min(metric.iteritems(), key=operator.itemgetter(1))[0]
