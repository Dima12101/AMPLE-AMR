"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from ample_amr.config import load_experiment_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    return project_root / "configs" / "warehouse_experiments.yaml"


@pytest.fixture(scope="session")
def experiment_config(config_path: Path):
    return load_experiment_config(config_path)
