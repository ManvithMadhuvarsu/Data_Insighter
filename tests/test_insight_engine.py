import pandas as pd

from insight_engine import funnel_insights, retention_cohort_insights, variance_explanation_insights


def test_funnel_insights_detect_weak_stage_conversion():
    df = pd.DataFrame({
        'visits': [100, 100],
        'signups': [50, 40],
        'purchase': [10, 5],
    })

    insights = funnel_insights(df, ['visits', 'signups', 'purchase'])

    assert insights
    assert insights[0]['kind'] == 'funnel'
    assert 'signups' in insights[0]['title']


def test_variance_explanation_identifies_segment_delta():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-02-01', '2026-02-02'] * 4),
        'segment': ['A', 'B', 'A', 'B'] * 4,
        'revenue': [100, 100, 200, 110] * 4,
    })

    insights = variance_explanation_insights(df, ['date'], ['segment'], ['revenue'])

    assert insights
    assert insights[0]['kind'] == 'variance_explanation'
    assert 'A' in insights[0]['detail']


def test_retention_cohort_insights_find_repeat_activity():
    df = pd.DataFrame({
        'date': pd.to_datetime([
            '2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05',
            '2026-02-01', '2026-02-02', '2026-02-03', '2026-02-04', '2026-02-05',
            '2026-03-01', '2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05',
            '2026-04-01', '2026-04-02', '2026-04-03', '2026-04-04', '2026-04-05',
        ]),
        'customer_id': [
            'c1', 'c2', 'c3', 'c4', 'c5',
            'c1', 'c2', 'c6', 'c7', 'c8',
            'c1', 'c3', 'c6', 'c9', 'c10',
            'c2', 'c4', 'c7', 'c9', 'c10',
        ],
    })

    insights = retention_cohort_insights(df, ['date'], ['customer_id'])

    assert insights
    assert insights[0]['kind'] == 'retention_cohort'
