import pandas as pd

from measure_service import evaluate_measure


def test_ratio_measure_uses_sum_of_columns():
    df = pd.DataFrame({
        'sales': [100, 200, 300],
        'orders': [10, 20, 30],
    })

    result = evaluate_measure(df, {
        'name': 'Average order value',
        'type': 'ratio',
        'numerator': 'sales',
        'denominator': 'orders',
    })

    assert result['value'] == 10
    assert result['type'] == 'ratio'


def test_margin_measure_returns_percentage():
    df = pd.DataFrame({
        'revenue': [1000, 500],
        'cost': [600, 300],
    })

    result = evaluate_measure(df, {
        'name': 'Gross margin',
        'type': 'margin',
        'revenue': 'revenue',
        'cost': 'cost',
    })

    assert result['value'] == 40


def test_period_change_compares_latest_period_to_previous():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2026-01-01', '2026-01-15', '2026-02-01', '2026-02-15']),
        'revenue': [100, 100, 150, 150],
    })

    result = evaluate_measure(df, {
        'name': 'MoM revenue',
        'type': 'period_change',
        'metric': 'revenue',
        'date_column': 'date',
        'grain': 'M',
    })

    assert result['value'] == 50
