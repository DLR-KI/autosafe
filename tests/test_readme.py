# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Executable-documentation guard: the README's Python examples run.

Extracts every ```python fenced block from README.md and executes
them sequentially in one shared namespace (later blocks may build on
earlier ones). If the public API drifts away from the README, this
test fails---the quickstart can never rot.
"""

import pathlib
import re

_README = pathlib.Path(__file__).parent.parent / "README.md"
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


class TestReadmeExamples:
    @staticmethod
    def test_python_blocks_execute_in_sequence():
        blocks = _BLOCK.findall(_README.read_text(encoding="utf-8"))
        assert len(blocks) >= 2, "README lost its quickstart examples"
        namespace: dict = {}
        for i, block in enumerate(blocks):
            code = compile(block, f"README.md#block{i}", "exec")
            exec(code, namespace)  # ruff:ignore[exec-builtin]
        # Block 0 exercised both heads of the API: construction from CSV
        # and an affinity query against a threshold.
        assert namespace["is_within_odd"].all()
        # Block 1 actually de-duplicated: iris carries three repeated
        # rows, so the policy must collapse 150 anchors to 147. Asserting
        # the count (not just that the call returned) is what keeps this
        # example honest if the dedup default ever flips.
        assert "policy" in namespace
        assert len(namespace["odd"].samples) == 147
