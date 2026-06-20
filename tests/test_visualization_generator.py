"""Unit tests for VisualizationGenerator chart rendering and the auto recommender.

These tests exercise the in-memory dataframe path so they stay fast and do not
touch the filesystem or the workspace store.
"""

import json

import numpy as np
import pandas as pd
import pytest

from visualization_generator import NumpyEncoder, VisualizationGenerator


@pytest.fixture()
def sales_df():
    rng = np.random.default_rng(42)
    periods = pd.date_range("2023-01-01", periods=60, freq="D")
    regions = rng.choice(["North", "South", "East", "West"], size=60)
    channels = rng.choice(["Web", "Retail", "Partner"], size=60)
    return pd.DataFrame(
        {
            "order_date": periods,
            "region": regions,
            "channel": channels,
            "revenue": rng.normal(1000, 250, size=60).round(2),
            "units": rng.integers(1, 50, size=60),
            "profit": rng.normal(200, 80, size=60).round(2),
        }
    )


def _figure(df, columns, viz_type, **kwargs):
    generator = VisualizationGenerator(dataframe=df)
    return generator.generate_visualization(columns, viz_type, **kwargs)


def test_bar_single_column(sales_df):
    fig = _figure(sales_df, ["region"], "bar")
    assert fig["data"], "bar chart should produce at least one trace"


def test_box_single_column_with_outliers():
    # Regression test: outlier filtering used to crash on operator precedence.
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200, -150]})
    fig = _figure(df, ["v"], "box")
    assert fig["data"]


def test_histogram_numeric(sales_df):
    fig = _figure(sales_df, ["revenue"], "histogram")
    assert fig["data"]


def test_scatter_requires_numeric(sales_df):
    fig = _figure(sales_df, ["revenue", "units"], "scatter")
    assert fig["data"]


def test_line_chart(sales_df):
    fig = _figure(sales_df, ["order_date", "revenue"], "line")
    assert fig["data"]


def test_datetime_chart_is_json_serializable(sales_df):
    # Regression: datetime axes used to break NumpyEncoder serialization.
    fig = _figure(sales_df, ["order_date", "revenue"], "line")
    json.dumps(fig, cls=NumpyEncoder)


@pytest.mark.parametrize(
    "columns,viz_type",
    [
        (["order_date", "revenue"], "area"),
        (["region", "revenue"], "donut"),
        (["region", "revenue"], "treemap"),
        (["channel", "units"], "funnel"),
        (["revenue", "units", "profit"], "bubble"),
    ],
)
def test_new_chart_types_render(sales_df, columns, viz_type):
    fig = _figure(sales_df, columns, viz_type)
    assert fig["data"]
    json.dumps(fig, cls=NumpyEncoder)


def test_auto_resolves_and_records_reason(sales_df):
    fig = _figure(sales_df, ["order_date", "revenue"], "auto")
    assert fig["layout"]["meta"]["auto_recommendation"]["reason"]


def test_duplicate_columns_do_not_crash(sales_df):
    fig = _figure(sales_df, ["revenue", "revenue"], "histogram")
    assert fig["data"]


def test_kpi_card(sales_df):
    fig = _figure(sales_df, ["revenue"], "kpi")
    assert fig["data"]


def test_unsupported_type_raises(sales_df):
    with pytest.raises(ValueError):
        _figure(sales_df, ["revenue"], "definitely_not_a_chart")
