# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Guards for the committed benchmark results tree.

The results under ``experiments/benchmark/results/`` are the artifacts
behind the paper's benchmark figures and tables. They are committed as
produced and are only regenerated together with the paper's numbers.
"""

from collections import Counter
from pathlib import Path

from experiments.benchmark import export_paper_data

RESULTS_DIR = export_paper_data.DEFAULT_RESULTS_DIR

#: Committed artifacts per suffix: one results.csv and config.json per
#: experiment (permutation stability writes a single JSON report), plus
#: the pgfplots-ready .dat files.
EXPECTED_SUFFIX_COUNTS = {".dat": 51, ".csv": 12, ".json": 14}


def test_results_tree_holds_exactly_the_audited_artifacts() -> None:
    """Fails if a result file appears, disappears, or a debug plot sneaks in."""
    files = [p for p in RESULTS_DIR.rglob("*") if p.is_file()]
    counts = Counter(p.suffix for p in files if p.name != "REUSE.toml")
    assert dict(counts) == EXPECTED_SUFFIX_COUNTS


def test_results_tree_rebuilds_every_paper_data_file(tmp_path: Path) -> None:
    status = export_paper_data.run(results_dir=RESULTS_DIR, outdir=tmp_path)
    assert sorted(status) == sorted(export_paper_data.EXPECTED_FILENAMES)
    assert all(s.startswith("written") for s in status.values()), status
