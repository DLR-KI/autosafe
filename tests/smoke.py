# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Run a subset of the tests as a smoke test."""

import numpy as np

import autosafe as af


def smoke_test() -> None:
    # Define anchor points as a samples list
    # The values are inspired by the Iris dataset.
    sample_list = [
        af.Sample(x=[5.1, 3.5, 1.4, 0.2]),
        af.Sample(x=[4.9, 3.0, 1.4, 0.2]),
        af.Sample(x=[4.7, 3.2, 1.3, 0.2]),
        af.Sample(x=[4.6, 3.1, 1.5, 0.2]),
        af.Sample(x=[5.0, 3.6, 1.4, 0.2]),
        af.Sample(x=[5.4, 3.9, 1.7, 0.4]),
        af.Sample(x=[4.6, 3.4, 1.4, 0.3]),
        af.Sample(x=[5.0, 3.4, 1.5, 0.2]),
        af.Sample(x=[4.4, 2.9, 1.4, 0.2]),
        af.Sample(x=[4.9, 3.1, 1.5, 0.1]),
        af.Sample(x=[5.4, 3.7, 1.5, 0.2]),
        af.Sample(x=[4.8, 3.4, 1.6, 0.2]),
        af.Sample(x=[4.8, 3.0, 1.4, 0.1]),
        af.Sample(x=[4.3, 3.0, 1.1, 0.1]),
        af.Sample(x=[5.8, 4.0, 1.2, 0.2]),
        af.Sample(x=[5.7, 4.4, 1.5, 0.4]),
        af.Sample(x=[5.4, 3.9, 1.3, 0.4]),
        af.Sample(x=[5.1, 3.5, 1.4, 0.3]),
        af.Sample(x=[5.7, 3.8, 1.7, 0.3]),
        af.Sample(x=[5.1, 3.8, 1.5, 0.3]),
    ]
    odd = af.Samples(
        samples=sample_list,
        closest_sample_mode="per_dimension",
        kernel_cls="RBF",
    )

    # Query the ODD for a new data point
    affinity = 0.8
    data_point = np.array([[5.1, 3.5, 1.4, 0.2]])
    is_within_odd = odd(data_point) >= affinity
    if all(is_within_odd) is False:
        raise AssertionError("Smoke test failed: data point is not within ODD.")


if __name__ == "__main__":
    smoke_test()
