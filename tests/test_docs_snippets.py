# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Executable-documentation guard: the rST docs' Python examples run.

Discovers every ``.rst`` file under ``docs/`` (excluding the generated
``docs/_build/`` tree, so a stale build never masquerades as a source
page), extracts every ``.. code-block:: python`` directive, and executes
each file's blocks sequentially in one shared namespace per file -- later
blocks may build on earlier ones, mirroring ``tests/test_readme.py``. A
newly added page is covered automatically: nothing here names a specific
file.

A block may opt out of execution via a ``.. no-execute: <reason>`` rST
comment on the line immediately above its directive. The single colon
keeps it a plain comment rather than a directive, so Sphinx never warns
about it. A reason is mandatory -- a skip with no stated reason is
exactly how docs rot unnoticed. Every block without that marker must
execute without raising.
"""

import dataclasses
import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_DOCS_ROOT = _REPO_ROOT / "docs"
_BUILD_DIR = _DOCS_ROOT / "_build"

_DIRECTIVE = re.compile(r"^\.\. code-block:: python[ \t]*$")
_NO_EXECUTE = re.compile(r"^\.\. no-execute:\s*(?P<reason>.*)$")

# A silent mass-deletion of examples (README-style rot) must fail the
# suite; this is comfortably below the current count so ordinary doc
# edits do not make the test flaky.
MIN_EXECUTED_BLOCKS = 4


@dataclasses.dataclass(frozen=True)
class DocBlock:
    """One extracted ``.. code-block:: python`` block.

    Attributes:
        source (str): Dedented Python source of the block's body.
        index (int): 0-based position of the block within its file,
            counting every ``code-block:: python`` directive including
            skipped ones (stable for error reporting).
        no_execute_reason (str | None): Reason text from a preceding
            ``.. no-execute:`` comment, or ``None`` if the block must
            execute.
    """

    source: str
    index: int
    no_execute_reason: str | None


def _line_indent(line: str) -> int:
    """Return a line's leading whitespace width.

    Args:
        line (str): One line of text, without its trailing newline.

    Returns:
        int: Number of leading whitespace characters.
    """
    return len(line) - len(line.lstrip())


def _dedent_body(
    lines: list[str], start: int, directive_indent: int
) -> tuple[str, int]:
    """Extract and dedent the body following a code-block directive.

    The body is the indented block of lines after the directive (and
    after any blank lines or ``:option:`` lines that precede the first
    content line); blank lines inside the body belong to it regardless
    of their own (lack of) indentation.

    Args:
        lines (list[str]): All lines of the file.
        start (int): Index of the first line after the directive line.
        directive_indent (int): Indentation column of the directive
            itself.

    Returns:
        tuple[str, int]: The dedented source text, and the index of the
            first line past the end of the block.
    """
    i, n = start, len(lines)

    while i < n and (
        not lines[i].strip()
        or (
            lines[i].lstrip().startswith(":")
            and _line_indent(lines[i]) > directive_indent
        )
    ):
        i += 1

    if i >= n or _line_indent(lines[i]) <= directive_indent:
        return "", i  # Directive with an empty body.

    body_indent = _line_indent(lines[i])
    body_lines: list[str] = []
    end = i
    while i < n:
        line = lines[i]
        if not line.strip():
            body_lines.append("")
            i += 1
            continue
        if _line_indent(line) < body_indent:
            break
        body_lines.append(line[body_indent:])
        i += 1
        end = i

    while body_lines and not body_lines[-1]:
        body_lines.pop()

    return "\n".join(body_lines), end


def _extract_blocks(text: str) -> list[DocBlock]:
    """Extract every ``.. code-block:: python`` block from rST source.

    Args:
        text (str): Full contents of one ``.rst`` file.

    Returns:
        list[DocBlock]: Blocks in document order.
    """
    lines = text.splitlines()
    blocks: list[DocBlock] = []
    index, i, n = 0, 0, len(lines)

    while i < n:
        if not _DIRECTIVE.match(lines[i].strip()):
            i += 1
            continue

        directive_indent = _line_indent(lines[i])

        no_execute_reason = None
        if i > 0:
            marker = _NO_EXECUTE.match(lines[i - 1].strip())
            if marker:
                reason = marker.group("reason").strip()
                assert reason, (
                    f"line {i}: '.. no-execute:' marker has no reason text -- "
                    "a skip without a stated reason is exactly how docs rot"
                )
                no_execute_reason = reason

        source, end = _dedent_body(lines, i + 1, directive_indent)
        blocks.append(
            DocBlock(source=source, index=index, no_execute_reason=no_execute_reason)
        )
        index += 1
        i = end

    return blocks


def _rst_files() -> list[pathlib.Path]:
    """Discover every ``.rst`` file under ``docs/``, excluding the build tree.

    Returns:
        list[pathlib.Path]: Sorted absolute paths.
    """
    return sorted(p for p in _DOCS_ROOT.rglob("*.rst") if _BUILD_DIR not in p.parents)


_FILES = _rst_files()
_IDS = [str(p.relative_to(_DOCS_ROOT)) for p in _FILES]


class TestDocsSnippets:
    @staticmethod
    @pytest.mark.parametrize("rst_path", _FILES, ids=_IDS)
    def test_python_blocks_execute_in_sequence(rst_path: pathlib.Path) -> None:
        blocks = _extract_blocks(rst_path.read_text(encoding="utf-8"))
        namespace: dict = {}
        for block in blocks:
            if block.no_execute_reason is not None:
                continue
            label = f"{rst_path.relative_to(_REPO_ROOT)} block {block.index}"
            code = compile(block.source, label, "exec")
            try:
                exec(code, namespace)  # ruff:ignore[exec-builtin]
            except Exception as exc:
                raise AssertionError(
                    f"{label} failed to execute ({type(exc).__name__}: {exc})."
                    f"\n--- source ---\n{block.source}"
                ) from exc

    @staticmethod
    def test_minimum_block_coverage():
        # Counts executed (non-skipped) blocks across every discovered
        # file, not just one -- so silently deleting all examples from
        # any page still fails the suite, matching the README guard.
        executed = sum(
            1
            for path in _FILES
            for block in _extract_blocks(path.read_text(encoding="utf-8"))
            if block.no_execute_reason is None
        )
        assert executed >= MIN_EXECUTED_BLOCKS, (
            f"Only {executed} executable 'code-block:: python' blocks found "
            f"under docs/ (floor={MIN_EXECUTED_BLOCKS}); did an example get "
            "deleted?"
        )
