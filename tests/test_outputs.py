"""Tests for result post-processing helpers."""

from __future__ import annotations

import pytest
import pandas as pd

from ample_amr.outputs import _add_relative_metrics


def test_add_relative_metrics_keeps_mode_profiles_separate() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "sensitivity_operation_modes",
                "scenario_size": "Warehouse-M",
                "seed": 0,
                "method": method,
                "mode_profile": profile,
                "social_welfare": welfare,
            }
            for profile, welfare_triplet in {
                "conservative": (10.0, 12.0, 13.0),
                "default": (20.0, 22.0, 25.0),
                "aggressive_bias": (30.0, 33.0, 39.0),
            }.items()
            for method, welfare in zip(
                ["fixed_heuristic", "fixed_auction", "ample_amr"],
                welfare_triplet,
                strict=True,
            )
        ]
    )

    result = _add_relative_metrics(frame)

    assert len(result) == len(frame)
    assert len(result[["scenario", "scenario_size", "seed", "method", "mode_profile"]].drop_duplicates()) == len(frame)

    conservative = result[(result["method"] == "ample_amr") & (result["mode_profile"] == "conservative")].iloc[0]
    default = result[(result["method"] == "ample_amr") & (result["mode_profile"] == "default")].iloc[0]
    aggressive = result[(result["method"] == "ample_amr") & (result["mode_profile"] == "aggressive_bias")].iloc[0]

    assert conservative["relative_welfare_gain_vs_fixed_heuristic"] == pytest.approx(0.3)
    assert default["relative_welfare_gain_vs_fixed_heuristic"] == pytest.approx(0.25)
    assert aggressive["relative_welfare_gain_vs_fixed_heuristic"] == pytest.approx(0.3)
