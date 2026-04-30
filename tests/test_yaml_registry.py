"""Tests for YAML-configurable forecaster registry."""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest

YAML_DIR = pathlib.Path(__file__).parent.parent


def _series(n: int = 60) -> pd.Series:
    t = np.arange(n)
    return pd.Series(
        100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12),
        index=pd.date_range("2020-01", periods=n, freq="ME"),
    )


def test_load_registry_from_yaml_bundled():
    """The bundled forecasters.yaml loads without error and includes core names."""
    from sktime_agentic import load_registry_from_yaml

    yaml_path = YAML_DIR / "forecasters.yaml"
    pytest.importorskip("yaml", reason="PyYAML not installed")

    reg = load_registry_from_yaml(yaml_path)
    assert "NaiveForecaster" in reg
    assert "MeanForecaster" in reg
    assert "SeasonalNaiveForecaster" in reg
    # Each entry must have cls, params, summary
    for name, meta in reg.items():
        assert callable(meta["cls"]), f"{name}: cls not callable"
        assert isinstance(meta["params"], dict)
        assert isinstance(meta["summary"], str)


def test_load_registry_missing_file():
    """FileNotFoundError raised for a non-existent path."""
    from sktime_agentic import load_registry_from_yaml

    pytest.importorskip("yaml", reason="PyYAML not installed")
    with pytest.raises(FileNotFoundError):
        load_registry_from_yaml("/tmp/does_not_exist_xyz.yaml")


def test_load_registry_skips_bad_imports(tmp_path):
    """Entries with unimportable modules are silently skipped."""
    from sktime_agentic import load_registry_from_yaml

    pytest.importorskip("yaml", reason="PyYAML not installed")
    yaml_content = textwrap.dedent("""\
        forecasters:
          - name: GoodForecaster
            module: sktime.forecasting.naive
            class: NaiveForecaster
            params: {strategy: last}
            summary: "Works fine."
          - name: BrokenForecaster
            module: totally.nonexistent.module
            class: DoesNotExist
            params: {}
            summary: "Should be skipped."
    """)
    config = tmp_path / "test.yaml"
    config.write_text(yaml_content)

    reg = load_registry_from_yaml(config)
    assert "GoodForecaster" in reg
    assert "BrokenForecaster" not in reg


def test_agentic_forecaster_uses_yaml_registry(tmp_path):
    """AgenticForecaster(registry_config=...) uses the YAML-defined forecasters."""
    pytest.importorskip("yaml", reason="PyYAML not installed")

    # Write a minimal YAML with only one forecaster
    yaml_content = textwrap.dedent("""\
        forecasters:
          - name: NaiveForecaster
            module: sktime.forecasting.naive
            class: NaiveForecaster
            params: {strategy: last}
            summary: "Last-value baseline."
          - name: MeanForecaster
            module: sktime.forecasting.naive
            class: NaiveForecaster
            params: {strategy: mean}
            summary: "Historical mean."
    """)
    config = tmp_path / "custom.yaml"
    config.write_text(yaml_content)

    from sktime_agentic import AgenticForecaster

    f = AgenticForecaster(
        prompt="test",
        backend="mock",
        holdout=12,
        registry_config=str(config),
    )
    f.fit(_series(), fh=list(range(1, 13)))

    assert f.selected_ in {"NaiveForecaster", "MeanForecaster"}
    y_pred = f.predict()
    assert len(y_pred) == 12
