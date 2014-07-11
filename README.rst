=====
PyDSE
=====

.. image:: https://travis-ci.org/blue-yonder/pydse.svg?branch=master
    :target: https://travis-ci.org/blue-yonder/pydse
.. image:: https://coveralls.io/repos/blue-yonder/pydse/badge.png
    :target: https://coveralls.io/r/blue-yonder/pydse
.. image:: https://requires.io/github/blue-yonder/pydse/requirements.png?branch=master
     :target: https://requires.io/github/blue-yonder/pydse/requirements/?branch=master
     :alt: Requirements Status

Toolset for Dynamic System Estimation for time series inspired by 
`DSE <http://cran.r-project.org/web/packages/dse/index.html>`_.
It is in a beta state and only includes ARMA models right now.
Documentation is available under http://pydse.readthedocs.org/.


Installation
============

To install in your home directory, use::

    python setup.py install --user

If your are using `virtualenv <http://virtualenv.readthedocs.org/en/latest/>`_,
just use `pip <http://pip.readthedocs.org/>`_::

    pip install pydse
