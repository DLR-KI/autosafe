<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>

SPDX-License-Identifier: CC-BY-SA-4.0
-->
# Experiment Configurations

This folder contains ready-to-run experiment files for:

- Monte Carlo ODD sampling from **2D to 12D**
- Dataset evaluation for **HCAS** and **VCAS**

## Structure

For each dimension `dim_Xd` (`X = 2..12`):

- `sampling_config.yaml`: unified Monte Carlo config
  (sampling + embedded `odd` section)

## Unified YAML config

`sampling_config.yaml` is a single config file that includes both:

- sampling settings (`dim`, limits, anchors, kernel, output filename)
- ODD definition (`odd:` block)

Run one experiment:

```bash
uv run autosafe montecarlo sample --config-file experiments/dim_2d/sampling_config.yaml
```

## Run all experiments

Use the experiments command with the provided batch spec:

```bash
uv run autosafe experiments run-spec experiments/run_all_spec.yaml
```

The spec performs:

1. Monte Carlo sampling for 2D..12D configs (``mode: mc-sample``)
2. ``autosafe evaluate dataset data/hcas_state_variables.csv``
3. ``autosafe evaluate dataset data/vcas_state_variables.csv``

## Notes

- ODD limits are configured for `[-10, 10]` ranges.
- Box limits are configured to exactly double ODD limits (`[-20, 20]`).
- ODD constraints contain only coupling inequalities (no duplicated box
  constraints), with one coupling constraint per dimension minus one.
- Output JSON files are written under `experiments/results/`.
