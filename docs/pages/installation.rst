.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

Installation
============

We recommend using `uv <https://astral.sh/uv/>`_ to manage the virtual
environment and dependencies. From the project root:

.. code-block:: console

    uv sync

This will create a virtual environment and install dependencies from `pyproject.toml`.

CVXOPT
------

Users might want to optionally install `CVXOPT <https://pypi.org/project/cvxopt/>`_ for enhanced performance.
However, as CVXOPT is licensed under GPL-v3, it is not included as a dependency by default.
Moreover, CVXOPT is not compatible with Python 3.14 or higher or any free-threaded builds as of now.
