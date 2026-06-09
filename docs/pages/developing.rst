.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

Developing
==========

We welcome contributions from the community to enhance and improve autoSAFE.

Prerequisites
-------------

First, make sure you cloned and installed the project as described in the `Installation <installation.html>`_ guide.

Pre-Commit Hooks
----------------

We use `prek <https://github.com/j178/prek>`_ to manage pre-commit hooks for code quality and consistency.
To install the pre-commit hooks, run the following command from the project root:

.. code-block:: console

    uv run prek install


Commits
-------

We strictly follow the `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_ specification for commit messages.
This means, commit messages have to start with one of the following types: ``build``, ``bump``, ``chore``, ``ci``, ``docs``, ``feat``, ``fix``, ``perf``, ``refactor``, ``revert``, ``style``, or ``test``.
Additionally, an optional scope can be provided in parentheses after the type, e.g., ``perf(ci): improve performance of CI pipeline``.

This is enforced by a pre-commit hook.

Moreover, we prefer signed commits.
However, this is currently not enforced, but might be at any time without prior notice.

Tests
-----

We use `pytest <https://pytest.org/>`_ for testing.
To run the test suite, execute the following command from the project root:

.. code-block:: console

    uv run pytest


VS Code
-------

The repository includes configuration files for Visual Studio Code (VS Code).
To set up VS Code for development, make sure to install the recommended extensions when prompted upon opening the project.
