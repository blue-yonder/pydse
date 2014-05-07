#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from numpy import linalg
from scipy import optimize


_logger = logging.getLogger(__name__)


class ARMAError(Exception):
    pass


class ARMA(object):
    """
        A(L)y(t) = B(L)e(t) + C(L)u(t) - TREND(t)

        L: Shift operator
        A: (axpxp) tensor to define auto-regression
        B: (bxpxp) tensor to define moving-average
        C: (cxpxm) tensor for external input
        e: (pxt) matrix of unobserved disturbance (white noise)
        y: (pxt) matrix of observed output variables
        u: (mxt) matrix of input variables
        TREND: (pxt) matrix like y or a p-dim vector
    """
    def __init__(self, A=None, B=None, C=None, TREND=None, rand_state=None):
        self.A = np.asarray(A[0]).reshape(A[1], order='F')
        self.B = np.asarray(B[0]).reshape(B[1], order='F') if B else np.zeros(shape=A[1])
        self.C = np.asarray(C[0]).reshape(C[1], order='F') if C else np.empty((0,))
        self.TREND = np.asarray(TREND) if TREND is not None else None
        self._check_consistency(self.A, self.B, self.C, self.TREND)

        self.Aconst = np.zeros(self.A.shape, dtype=np.bool)
        self.Bconst = np.zeros(self.B.shape, dtype=np.bool)
        self.Cconst = np.zeros(self.C.shape, dtype=np.bool)

        self.rand = rand_state if rand_state is not None else np.random.RandomState()

    def _set_array_by_mask(self, arr, mask, values):
        mask = np.where(mask == False)
        arr[mask] = values

    def _get_array_by_mask(self, arr, mask):
        mask = np.where(mask == False)
        return arr[mask]

    def _get_num_non_consts(self):
        a = np.sum(self.Aconst == False)
        b = np.sum(self.Bconst == False)
        c = np.sum(self.Cconst == False)
        return (a, b, c)

    @property
    def non_consts(self):
        a, b, c = self._get_num_non_consts()
        A = self._get_array_by_mask(self.A, self.Aconst)
        B = self._get_array_by_mask(self.B, self.Bconst)
        C = self._get_array_by_mask(self.C, self.Cconst)
        return np.hstack([A, B, C])

    @non_consts.setter
    def non_consts(self, values):
        a, b, c = self._get_num_non_consts()
        if values.size != a + b + c:
            raise ARMAError("Number of values does not equal number of non-constants")
        A_values = values[:a]
        B_values = values[a:a + b]
        C_values = values[a + b:a + b + c]
        self._set_array_by_mask(self.A, self.Aconst, A_values)
        self._set_array_by_mask(self.B, self.Bconst, B_values)
        self._set_array_by_mask(self.C, self.Cconst, C_values)

    def _check_consistency(self, A, B, C, TREND):
        if A is None:
            raise ARMAError("A needs to be set for an ARMA model")
        n = A.shape[1]
        if n != A.shape[2] or len(A.shape) > 3:
            raise ARMAError("A needs to be of shape (a, p, p)")
        if n != B.shape[1] or (n != B.shape[2] or len(B.shape) > 3):
            raise ARMAError("B needs to be of shape (b, p, p) with A being of shape (a, p, p)")
        if C.size != 0 and (n != C.shape[1] or len(C.shape) > 3):
            raise ARMAError("C needs to be of shape (c, p, m) with A being of shape (a, p, p)")
        if TREND is not None:
            if len(TREND.shape) > 2:
                raise ARMAError("TREND needs to of shape (p, t) with A being of shape (a, p, p)")
            elif len(TREND.shape) == 2 and  n != TREND.shape[0]:
                raise ARMAError("TREND needs to of shape (p, t) with A being of shape (a, p, p)")
            elif len(TREND.shape) == 1 and n != TREND.shape[0]:
                raise ARMAError("TREND needs to of shape (p, t) with A being of shape (a, p, p)")

    def _get_noise(self, samples, p, lags):
        w0 = self.rand.normal(size=lags * p).reshape((lags, p))
        w = self.rand.normal(size=samples * p).reshape((samples, p))
        return (w0, w)

    def _prep_y(self, trend, dim_t, dim_p):
        if trend is not None:
            if len(trend.shape) == 2:
                assert trend.shape[1] == dim_t
                y = np.copy(trend)
            else:
                y = np.tile(trend, (dim_t, 1))
        else:
            y = np.zeros((dim_t, dim_p))
        return y

    def simulate(self, y0=None, u0=None, sampleT=100, noise=None):
        p = self.A.shape[1]
        a, b = self.A.shape[0], self.B.shape[0]
        c = self.C.shape[0] if self.C else 0
        m = self.C.shape[2] if self.C else 0
        y0 = y0 if y0 else np.zeros((a, p))
        u0 = u0 if u0 else np.zeros((c, m))

        # generate white noise if necessary
        if not noise:
            noise = self._get_noise(sampleT, p, b)
        w0, w = noise

        # diagonalize with respect to matrix of leading coefficients
        A0inv = linalg.inv(self.A[0, :, :])
        A = np.tensordot(self.A, A0inv, axes=1)
        B = np.tensordot(self.B, A0inv, axes=1)
        if self.C:
            C = np.tensordot(self.C, A0inv, axes=1)

        # perform simulation
        y = self._prep_y(self.TREND, sampleT, p)
        for t in xrange(sampleT):
            for l in xrange(1, a):
                if t - l <= -1:
                    y[t, :] = y[t, :] - np.dot(A[l, :, :], y0[l - t - 1, :])
                else:
                    y[t, :] = y[t, :] - np.dot(A[l, :, :], y[t - l, :])

            for l in xrange(b):
                if t - l <= -1:
                    y[t, :] = y[t, :] + np.dot(B[l, :, :], w0[l - t - 1, :])
                else:
                    y[t, :] = y[t, :] + np.dot(B[l, :, :], w[t - l, :])

            for l in xrange(c):
                if t - l <= -1:
                    y[t, :] = y[t, :] + np.dot(C[l, :, :], u0[l - t - 1, :])
                else:
                    y[t, :] = y[t, :] + np.dot(C[l, :, :], u[t - l, :])

        return y

    def forecast(self, y, u=None):
        p = self.A.shape[1]
        a, b = self.A.shape[0], self.B.shape[0]
        c = self.C.shape[0] if self.C else 0
        m = self.C.shape[2] if self.C else 0
        TREND = self.TREND

        # ToDo: Let these be parameters and do consistensy check
        sampleT = predictT = y.shape[0]
        pred_err = np.zeros((sampleT, p))

        if TREND is not None:
            if len(TREND.shape) == 2:
                assert TREND.shape[1] == sampleT
            else:
                TREND = np.tile(self.TREND, (sampleT, 1))

        # diagonalize with respect to matrix of leading coefficients
        B0inv = linalg.inv(self.B[0, :, :])
        A = np.tensordot(self.A, B0inv, axes=1)
        B = np.tensordot(self.B, B0inv, axes=1)
        if self.C:
            C = np.tensordot(self.C, B0inv, axes=1)
        if TREND is not None:
            TREND = np.dot(TREND, B0inv)

        # perform prediction
        for t in xrange(sampleT):
            if TREND is not None:
                vt = -TREND[t, :]
            else:
                vt = np.zeros((p,))

            for l in xrange(a):
                if l <= t:
                    vt = vt + np.dot(A[l, :, :], y[t - l, :])

            for l in xrange(1, b):
                if l <= t:
                    vt = vt - np.dot(B[l, :, :], pred_err[t - l, :])

            for l in xrange(c):
                if l <= t:
                    vt = vt - np.dot(C[l, :, :], u[t - l, :])

            pred_err[t, :] = vt

        pred = np.zeros((predictT, p))
        pred[:sampleT, :] = y[:sampleT, :] - np.dot(pred_err, B[0, :, :])

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

    @staticmethod
    def negloglike(pred, y):
        sampleT = pred.shape[0]
        res = pred[:sampleT, :] - y[:sampleT, :]
        p = res.shape[1]

        Om = np.dot(res.T, res) / sampleT

        if np.any(np.isnan(Om)) or np.any(Om > 1e100):
            like1 = like2 = 1e100
        else:
            _, s, _ = linalg.svd(Om)

            # Check for degeneracy
            non_degen_mask = s > s[0] * np.sqrt(np.finfo(np.float).eps)
            if not np.all(non_degen_mask):
                _logger.warn("Covariance matrix is singular. Working on subspace")
                s = s[non_degen_mask]

            like1 = 0.5 * sampleT * np.log(np.prod(s))
            like2 = 0.5 * sampleT * len(s)

        const = 0.5 * sampleT * p * np.log(2 * np.pi)
        return like1 + like2 + const

    def est_params(self, y):
        def cost_function(x):
            self.non_consts = x
            pred = self.forecast(y=y)
            return self.negloglike(pred, y)

        x0 = self.non_consts
        return optimize.minimize(cost_function, x0)

