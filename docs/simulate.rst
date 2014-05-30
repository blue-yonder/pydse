Create and Simulate
===================

The definition of an ARMA model is:

.. math::

    A(L)y_t = B(L)e_t + C(L)u_t

where :math:`L` is the *lag* operator, :math:`y_t` a :math:`p`-dimensional
vector of observed output variables, :math:`e_t` a :math:`p`-dimensional vector
of white noise and :math:`u_t` a :math:`m`-dimensional vector of input variables.
Since :math:`A, B` and :math:`C` are matrices in the lag shift operator, we have
:math:`A(L)` is a :math:`a \times\, p \times\, p` tensor to define auto-regression,
:math:`B(L)` is a :math:`b \times\, p \times\, p` tensor to moving-average and
:math:`C(L)` is a :math:`c \times\, p \times\, m` tensor to account for the input
variables.

We create a simple ARMA model for a two dimensional output vector with matrices:

.. math::

    A(L) = \left( \begin{array}{cc}
    1+0.5L1+0.3L2 & 0+0.2L1+0.1L2\\
    0+0.2L1+0.05L2 & 1+0.5L1+0.3L2\end{array} \right),

.. math::

    B(L) =\left( \begin{array}{cc}
    1+0.2L1 & 0+0.1L1\\
    0+0.0L1 & 1+0.3L1\end{array} \right)

In order to set this matrix we just write the entries left to right, up to down
into an array and define the shape of this array in a second array:

.. plot::
    :include-source:
    :context:
    :nofigs:

    import pandas as pd
    import numpy as np
    import matplotlib.pylab as plt
    from pydse.arma import ARMA

    AR = (np.array([1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3]), np.array([3, 2, 2]))
    MA = (np.array([1, .2, 0, .1, 0, 0, 1, .3]), np.array([2, 2, 2]))
    arma = ARMA(A=AR, B=MA, rand_state=0)

Note that we set the random state to seed 0 to get the same results.
Then by simulating we get:

.. plot::
    :include-source:
    :context:

    sim_data = arma.simulate(sampleT=100)
    sim_index = pd.date_range('1/1/2011', periods=sim_data.shape[0], freq='d')
    df = pd.DataFrame(data=sim_data, index=sim_index)
    df.plot()

.. plot::
    :context:

    plt.close()

Let's create a simpler ARMA model with scalar output variable.

.. plot::
    :include-source:
    :context:

    AR = (np.array([1, .5, .3]), np.array([3, 1, 1]))
    MA = (np.array([1, .2]), np.array([2, 1, 1]))
    arma = ARMA(A=AR, B=MA, rand_state=0)

Quite often you wanna check the `autocorrelation function <http://en.wikipedia.org/wiki/Autocorrelation_function>`__
and `partial autocorrelation function <http://en.wikipedia.org/wiki/Partial_autocorrelation_function>`__:

.. plot::
    :include-source:
    :context:

    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

    sim_data = arma.simulate(sampleT=3000)
    sim_index = pd.date_range('1/1/2011', periods=sim_data.shape[0], freq='d')
    df = pd.DataFrame(data=sim_data, index=sim_index)
    plot_acf(df[0], lags=10)
    plot_pacf(df[0], lags=10)

Find a good introduction to ARMA on the `Decision 411 <http://people.duke.edu/~rnau/Decision411CoursePage.htm>`__
course page.
