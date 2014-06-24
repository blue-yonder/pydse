# -*- encoding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from numpy import linalg

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"

_logger = logging.getLogger(__name__)


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
            _logger.warn("Covariance matrix is singular. "
                         "Working on subspace.")
            s = s[non_degen_mask]

        like1 = 0.5 * sampleT * np.log(np.prod(s))
        like2 = 0.5 * sampleT * len(s)

    const = 0.5 * sampleT * p * np.log(2 * np.pi)
    return like1 + like2 + const