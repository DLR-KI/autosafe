.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

.. image:: _static/autoSAFE.svg
    :align: center
    :width: 300px

Description
-----------

Reference implementation of the autoSAFE specification.

autoSAFE can derive semantically correct Operational Design Domains (ODDs) purely from data.
This allows to automatically generate ODDs for machine learning-based functions.
The use cases range from data-driven ODD definition over ODD monitoring to retrofitting existing functions with ODDs.
Moreover, if no ODD is given for a certain dataset, autoSAFE can derive one automatically to ensure safe operations of the resulting AI-based system.

General Information
-------------------

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    pages/installation
    pages/usage
    pages/tools
    pages/comparison_methods
    pages/developing

.. toctree::
    :maxdepth: 1
    :caption: Python API Reference:

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
