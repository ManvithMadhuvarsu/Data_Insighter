from typing import Any, Dict, List

import numpy as np
import pandas as pd


def build_explanations(insight: Dict[str, Any], why: List[str]) -> Dict[str, Any]:
    enriched = dict(insight)
    enriched['why_this_matters'] = why
    return enriched


def anomaly_insights(df: pd.DataFrame, numeric_columns: List[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for column in numeric_columns[:8]:
        series = df[column].dropna().astype(float)
        if len(series) < 10:
            continue
        z_scores = ((series - series.mean()) / series.std(ddof=0)).replace([np.inf, -np.inf], np.nan)
        outliers = series[z_scores.abs() >= 2.5].sort_values(ascending=False)
        if outliers.empty:
            continue

        top_value = float(outliers.iloc[0])
        findings.append(build_explanations({
            'kind': 'anomaly',
            'title': f'{column} shows unusually extreme values',
            'detail': f'The strongest outlier observed is {top_value:,.2f}, far from the typical range.',
            'score': round(float(z_scores.abs().max()), 3),
            'stat': f'{len(outliers)} outliers (z >= 2.5)',
            'recommended_chart': {
                'title': f'{column} anomaly scan',
                'description': 'Use a box plot or histogram to inspect extreme points before trusting aggregate results.',
                'type': 'box',
                'columns': [column],
                'sample_percentage': 100,
            }
        }, [
            'Outlier detection is based on z-scores compared with the rest of the observed distribution.',
            'Extreme values can distort averages, forecasts, and model assumptions.',
        ]))

    findings.sort(key=lambda item: item['score'], reverse=True)
    return findings[:2]


def segment_driver_insights(df: pd.DataFrame, dimensions: List[str], measures: List[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for dimension in dimensions[:6]:
        cardinality = df[dimension].nunique(dropna=True)
        if cardinality < 2 or cardinality > 12:
            continue

        for measure in measures[:6]:
            grouped = df[[dimension, measure]].dropna()
            if grouped.empty or grouped[measure].nunique() < 2:
                continue

            summary = grouped.groupby(dimension)[measure].mean().sort_values(ascending=False)
            if len(summary) < 2:
                continue

            top_segment = summary.index[0]
            bottom_segment = summary.index[-1]
            spread = float(summary.iloc[0] - summary.iloc[-1])
            if spread == 0:
                continue

            findings.append(build_explanations({
                'kind': 'segment_driver',
                'title': f'{dimension} strongly separates {measure}',
                'detail': f"{top_segment} leads while {bottom_segment} trails, with an average gap of {spread:,.2f}.",
                'score': round(abs(spread), 3),
                'stat': f'{top_segment} vs {bottom_segment}',
                'recommended_chart': {
                    'title': f'{measure} by {dimension}',
                    'description': 'Rank segments to see which groups are driving the measure up or down.',
                    'type': 'bar',
                    'columns': [dimension, measure],
                    'sample_percentage': 100,
                }
            }, [
                'The measure was aggregated by segment and compared using average values.',
                'Large separation between segments often points to business drivers worth investigating.',
            ]))

    findings.sort(key=lambda item: item['score'], reverse=True)
    return findings[:3]


def contribution_insights(df: pd.DataFrame, dimensions: List[str], measures: List[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for dimension in dimensions[:6]:
        cardinality = df[dimension].nunique(dropna=True)
        if cardinality < 2 or cardinality > 12:
            continue

        for measure in measures[:6]:
            grouped = df[[dimension, measure]].dropna()
            if grouped.empty:
                continue

            contribution = grouped.groupby(dimension)[measure].sum().sort_values(ascending=False)
            total = float(contribution.sum())
            if total == 0 or len(contribution) < 2:
                continue

            top_segment = contribution.index[0]
            top_share = float(contribution.iloc[0] / total * 100)
            if top_share < 35:
                continue

            findings.append(build_explanations({
                'kind': 'contribution',
                'title': f'{top_segment} dominates contribution to {measure}',
                'detail': f'{top_segment} contributes {top_share:.2f}% of the total {measure}.',
                'score': round(top_share, 3),
                'stat': f'{top_share:.2f}% share',
                'recommended_chart': {
                    'title': f'{measure} contribution by {dimension}',
                    'description': 'Use a ranked view to understand which segments contribute the most to the total.',
                    'type': 'bar',
                    'columns': [dimension, measure],
                    'sample_percentage': 100,
                }
            }, [
                'Contribution was measured using the sum of the selected metric by segment.',
                'A single segment dominating the total can indicate concentration risk or a growth opportunity.',
            ]))

    findings.sort(key=lambda item: item['score'], reverse=True)
    return findings[:2]


def seasonality_insights(df: pd.DataFrame, datetime_columns: List[str], measures: List[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not datetime_columns or not measures:
        return findings

    for date_column in datetime_columns[:2]:
        for measure in measures[:4]:
            working = df[[date_column, measure]].dropna()
            if len(working) < 12:
                continue

            working = working.copy()
            working['month_name'] = pd.to_datetime(working[date_column], errors='coerce').dt.month_name()
            monthly = working.groupby('month_name')[measure].mean().dropna()
            if monthly.nunique() < 2 or len(monthly) < 3:
                continue

            peak = monthly.idxmax()
            trough = monthly.idxmin()
            spread = float(monthly.max() - monthly.min())
            findings.append(build_explanations({
                'kind': 'seasonality',
                'title': f'{measure} shows a repeating month pattern',
                'detail': f'{peak} tends to be strongest while {trough} is weakest based on average observed values.',
                'score': round(abs(spread), 3),
                'stat': f'{peak} vs {trough}',
                'recommended_chart': {
                    'title': f'{measure} over time',
                    'description': 'Inspect the time series to see whether a repeating cycle or seasonal pattern is present.',
                    'type': 'line',
                    'columns': [date_column, measure],
                    'sample_percentage': 100,
                }
            }, [
                'The engine grouped the metric by month name after parsing the date field.',
                'Seasonal shape matters when comparing periods, forecasting, or explaining spikes.',
            ]))

    findings.sort(key=lambda item: item['score'], reverse=True)
    return findings[:1]


def variance_explanation_insights(
    df: pd.DataFrame,
    datetime_columns: List[str],
    dimensions: List[str],
    measures: List[str],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not datetime_columns or not dimensions or not measures:
        return findings

    for date_column in datetime_columns[:1]:
        for measure in measures[:4]:
            working = df[[date_column, measure] + dimensions[:4]].copy()
            working[date_column] = pd.to_datetime(working[date_column], errors='coerce')
            working[measure] = pd.to_numeric(working[measure], errors='coerce')
            working = working.dropna(subset=[date_column, measure])
            if len(working) < 12:
                continue

            working['period'] = working[date_column].dt.to_period('M')
            period_totals = working.groupby('period')[measure].sum().sort_index()
            if len(period_totals) < 2:
                continue

            previous_period = period_totals.index[-2]
            latest_period = period_totals.index[-1]
            total_delta = float(period_totals.iloc[-1] - period_totals.iloc[-2])
            if total_delta == 0:
                continue

            for dimension in dimensions[:4]:
                segment_periods = working.groupby(['period', dimension])[measure].sum().unstack(fill_value=0)
                if previous_period not in segment_periods.index or latest_period not in segment_periods.index:
                    continue
                deltas = (segment_periods.loc[latest_period] - segment_periods.loc[previous_period]).sort_values(
                    key=lambda values: values.abs(),
                    ascending=False,
                )
                if deltas.empty:
                    continue
                top_segment = deltas.index[0]
                top_delta = float(deltas.iloc[0])
                share = abs(top_delta / total_delta * 100) if total_delta else 0
                findings.append(build_explanations({
                    'kind': 'variance_explanation',
                    'title': f'{dimension} explains the latest change in {measure}',
                    'detail': f"{top_segment} contributed {top_delta:,.2f} of the latest period change ({share:.1f}% of net movement).",
                    'score': round(abs(top_delta), 3),
                    'stat': f'{latest_period} vs {previous_period}',
                    'recommended_chart': {
                        'title': f'{measure} variance by {dimension}',
                        'description': 'Compare segment movement between recent periods to explain what changed.',
                        'type': 'bar',
                        'columns': [dimension, measure],
                        'sample_percentage': 100,
                    }
                }, [
                    'The engine compared the latest monthly total with the previous monthly total.',
                    'Segment deltas identify which group most explains the net movement.',
                ]))

    findings.sort(key=lambda item: item['score'], reverse=True)
    return findings[:2]


def funnel_insights(df: pd.DataFrame, measures: List[str]) -> List[Dict[str, Any]]:
    stage_order = ['visit', 'view', 'lead', 'signup', 'trial', 'cart', 'checkout', 'purchase', 'paid']
    matched = []
    for column in measures:
        name = column.lower()
        for idx, token in enumerate(stage_order):
            if token in name:
                matched.append((idx, column))
                break

    matched = sorted(matched)
    if len(matched) < 2:
        return []

    stages = [column for _, column in matched[:5]]
    totals = [(column, float(pd.to_numeric(df[column], errors='coerce').sum())) for column in stages]
    drops = []
    for (from_col, from_total), (to_col, to_total) in zip(totals, totals[1:]):
        if from_total <= 0:
            continue
        conversion = to_total / from_total * 100
        drops.append((from_col, to_col, conversion, from_total - to_total))

    if not drops:
        return []

    weakest = sorted(drops, key=lambda item: item[2])[0]
    return [build_explanations({
        'kind': 'funnel',
        'title': f'Largest funnel friction appears between {weakest[0]} and {weakest[1]}',
        'detail': f'The observed conversion between these stages is {weakest[2]:.2f}%, with a drop of {weakest[3]:,.2f}.',
        'score': round(100 - weakest[2], 3),
        'stat': f'{weakest[2]:.2f}% conversion',
        'recommended_chart': {
            'title': 'Funnel stage totals',
            'description': 'Review stage totals to see where the funnel loses the most volume.',
            'type': 'bar',
            'columns': stages[:2],
            'sample_percentage': 100,
        }
    }, [
        'Funnel stages were inferred from common stage names in numeric columns.',
        'The weakest stage-to-stage conversion is usually the first place to investigate process friction.',
    ])]


def retention_cohort_insights(
    df: pd.DataFrame,
    datetime_columns: List[str],
    identifier_columns: List[str],
) -> List[Dict[str, Any]]:
    if not datetime_columns or not identifier_columns:
        return []

    date_column = datetime_columns[0]
    id_column = identifier_columns[0]
    working = df[[date_column, id_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors='coerce')
    working[id_column] = working[id_column].astype(str)
    working = working.dropna(subset=[date_column, id_column])
    if working[id_column].nunique() < 10 or len(working) < 20:
        return []

    working['period'] = working[date_column].dt.to_period('M')
    first_period = working.groupby(id_column)['period'].min()
    working = working.join(first_period.rename('cohort'), on=id_column)
    cohort_sizes = working.groupby('cohort')[id_column].nunique()
    repeat_counts = working[working['period'] > working['cohort']].groupby('cohort')[id_column].nunique()
    retention = (repeat_counts / cohort_sizes * 100).dropna()
    if retention.empty:
        return []

    strongest = retention.sort_values(ascending=False).index[0]
    value = float(retention.loc[strongest])
    return [build_explanations({
        'kind': 'retention_cohort',
        'title': f'{strongest} cohort has the strongest repeat activity',
        'detail': f'{value:.2f}% of entities in that cohort appear again in a later period.',
        'score': round(value, 3),
        'stat': f'{value:.2f}% repeat rate',
        'recommended_chart': {
            'title': f'{id_column} cohort retention',
            'description': 'Inspect repeat activity by cohort to understand retention-style behavior.',
            'type': 'heatmap',
            'columns': [date_column, id_column],
            'sample_percentage': 100,
        }
    }, [
        'Cohorts were inferred from each identifier\'s first observed month.',
        'Repeat activity is estimated by checking whether an entity appears in later months.',
    ])]


def build_executive_takeaways(insights: List[Dict[str, Any]], quality_alerts: List[Dict[str, Any]]) -> List[str]:
    takeaways: List[str] = []
    if quality_alerts:
        top_alert = quality_alerts[0]
        takeaways.append(f"Data quality note: {top_alert['title']} - {top_alert['detail']}")

    for insight in insights[:3]:
        takeaways.append(f"{insight['title']}: {insight['detail']}")

    return takeaways[:4]
