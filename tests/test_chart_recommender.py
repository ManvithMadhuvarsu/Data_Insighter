"""Unit tests for the automatic chart recommender.

These are pure-logic tests (no plotly figure rendering), so they run fast.
"""

import numpy as np
import pandas as pd
import pytest

from chart_recommender import recommend_chart


@pytest.fixture()
def df():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "order_date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "region": rng.choice(["North", "South", "East", "West"], size=50),
            "channel": rng.choice(["Web", "Retail", "Partner"], size=50),
            "revenue": rng.normal(1000, 200, size=50).round(2),
            "units": rng.integers(1, 40, size=50),
        }
    )


def test_single_measure_recommends_histogram(df):
    assert recommend_chart(df, ["revenue"])["type"] == "histogram"


def test_single_dimension_recommends_bar(df):
    assert recommend_chart(df, ["region"])["type"] == "bar"


def test_date_plus_measure_recommends_line(df):
    rec = recommend_chart(df, ["order_date", "revenue"])
    assert rec["type"] == "line"
    # The date should be ordered first so it lands on the x-axis.
    assert rec["columns"][0] == "order_date"


def test_two_measures_recommend_scatter(df):
    assert recommend_chart(df, ["revenue", "units"])["type"] == "scatter"


def test_dimension_plus_measure_recommends_bar(df):
    rec = recommend_chart(df, ["region", "revenue"])
    assert rec["type"] == "bar"
    assert rec["columns"][0] == "region"


def test_two_dimensions_recommend_heatmap(df):
    assert recommend_chart(df, ["region", "channel"])["type"] == "heatmap"


def test_measure_independent_of_selection_order(df):
    # A measure selected before a date should still produce a time-series line.
    rec = recommend_chart(df, ["revenue", "order_date"])
    assert rec["type"] == "line"
    assert rec["columns"][0] == "order_date"


def test_empty_selection_raises(df):
    with pytest.raises(ValueError):
        recommend_chart(df, [])


def test_recommendation_includes_reason(df):
    assert recommend_chart(df, ["revenue"])["reason"]
