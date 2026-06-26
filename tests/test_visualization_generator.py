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


@pytest.fixture()
def wide_df():
    rng = np.random.default_rng(1)
    n = 600
    return pd.DataFrame(
        {
            "order_date": np.repeat(pd.date_range("2023-01-01", periods=30, freq="D"), 20),
            "customer": [f"cust_{i % 80}" for i in range(n)],
            "region": rng.choice(["North", "South", "East", "West"], size=n),
            "revenue": rng.normal(1000, 200, size=n).round(2),
        }
    )


def test_line_aggregates_duplicate_x(wide_df):
    # 600 rows across 30 dates must collapse to 30 chronological points.
    fig = _figure(wide_df, ["order_date", "revenue"], "line")
    xs = list(fig["data"][0]["x"])
    assert len(xs) == 30
    assert xs == sorted(xs)


def test_bar_caps_and_sorts_high_cardinality(wide_df):
    fig = _figure(wide_df, ["customer", "revenue"], "bar")
    ys = list(fig["data"][0]["y"])
    assert len(ys) <= 25
    assert ys == sorted(ys, reverse=True)


def test_pie_buckets_into_other_and_preserves_total(wide_df):
    fig = _figure(wide_df, ["customer", "revenue"], "pie")
    labels = list(fig["data"][0]["labels"])
    values = list(fig["data"][0]["values"])
    assert "Other" in labels
    assert len(labels) <= 13
    assert abs(sum(values) - wide_df["revenue"].sum()) < 1e-6


def test_line_without_numeric_y_raises(wide_df):
    with pytest.raises(ValueError):
        _figure(wide_df, ["order_date", "region"], "line")


@pytest.mark.parametrize("viz_type", ["line", "area"])
def test_time_series_with_missing_dates_serialize(viz_type):
    # Regression: a NaT in the date axis crashed serialization
    # ("NaTType does not support strftime"). Rows are unique so no
    # aggregation masks the NaT.
    df = pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", None, "2023-01-04", "2023-01-05"]
            ),
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    fig = _figure(df, ["order_date", "revenue"], viz_type)
    json.dumps(fig, cls=NumpyEncoder)
    # The undated row is dropped, leaving four points.
    assert len(fig["data"][0]["x"]) == 4


def test_kpi_card(sales_df):
    fig = _figure(sales_df, ["revenue"], "kpi")
    assert fig["data"]


def test_unsupported_type_raises(sales_df):
    with pytest.raises(ValueError):
        _figure(sales_df, ["revenue"], "definitely_not_a_chart")
