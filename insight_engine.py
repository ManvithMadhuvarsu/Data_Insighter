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


def build_executive_takeaways(insights: List[Dict[str, Any]], quality_alerts: List[Dict[str, Any]]) -> List[str]:
    takeaways: List[str] = []
    if quality_alerts:
        top_alert = quality_alerts[0]
        takeaways.append(f"Data quality note: {top_alert['title']} - {top_alert['detail']}")

    for insight in insights[:3]:
        takeaways.append(f"{insight['title']}: {insight['detail']}")

    return takeaways[:4]

