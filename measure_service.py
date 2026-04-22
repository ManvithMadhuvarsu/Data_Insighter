from typing import Any, Dict

import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if not column or column not in df.columns:
        raise ValueError(f"Column '{column}' was not found.")
    return pd.to_numeric(df[column], errors='coerce')


def _datetime(df: pd.DataFrame, column: str) -> pd.Series:
    if not column or column not in df.columns:
        raise ValueError(f"Date column '{column}' was not found.")
    dates = pd.to_datetime(df[column], errors='coerce')
    if dates.notna().sum() == 0:
        raise ValueError(f"Column '{column}' could not be interpreted as dates.")
    return dates


def evaluate_measure(df: pd.DataFrame, definition: Dict[str, Any]) -> Dict[str, Any]:
    measure_type = definition.get('type')
    name = definition.get('name') or measure_type or 'Measure'

    if measure_type == 'ratio':
        numerator = _numeric(df, definition.get('numerator')).sum()
        denominator = _numeric(df, definition.get('denominator')).sum()
        value = float(numerator / denominator) if denominator else np.nan
        explanation = 'Ratio uses the sum of numerator divided by the sum of denominator.'

    elif measure_type == 'margin':
        revenue = _numeric(df, definition.get('revenue')).sum()
        cost = _numeric(df, definition.get('cost')).sum()
        value = float(((revenue - cost) / revenue) * 100) if revenue else np.nan
        explanation = 'Margin uses (revenue - cost) divided by revenue, expressed as a percentage.'

    elif measure_type == 'conversion_rate':
        converted = _numeric(df, definition.get('converted')).sum()
        total = _numeric(df, definition.get('total')).sum()
        value = float((converted / total) * 100) if total else np.nan
        explanation = 'Conversion rate uses converted count divided by total count, expressed as a percentage.'

    elif measure_type == 'growth_pct':
        metric = definition.get('metric')
        date_column = definition.get('date_column')
        working = pd.DataFrame({
            'date': _datetime(df, date_column),
            'metric': _numeric(df, metric),
        }).dropna().sort_values('date')
        if len(working) < 2:
            raise ValueError('Growth percentage requires at least two dated metric values.')
        first = float(working['metric'].iloc[0])
        last = float(working['metric'].iloc[-1])
        value = float(((last - first) / abs(first)) * 100) if first else np.nan
        explanation = 'Growth percentage compares the earliest and latest observed metric values.'

    elif measure_type == 'rolling_average':
        metric = definition.get('metric')
        date_column = definition.get('date_column')
        window = int(definition.get('window') or 7)
        working = pd.DataFrame({
            'date': _datetime(df, date_column),
            'metric': _numeric(df, metric),
        }).dropna().sort_values('date')
        if working.empty:
            raise ValueError('Rolling average requires usable dated metric values.')
        rolling = working['metric'].rolling(window=min(window, len(working))).mean()
        value = float(rolling.dropna().iloc[-1])
        explanation = f'Rolling average uses the latest {window}-row moving average after sorting by date.'

    elif measure_type == 'period_change':
        metric = definition.get('metric')
        date_column = definition.get('date_column')
        grain = definition.get('grain') or 'M'
        if grain not in {'M', 'Q', 'Y'}:
            raise ValueError('Period change grain must be M, Q, or Y.')
        working = pd.DataFrame({
            'date': _datetime(df, date_column),
            'metric': _numeric(df, metric),
        }).dropna()
        period_totals = working.groupby(working['date'].dt.to_period(grain))['metric'].sum().sort_index()
        if len(period_totals) < 2:
            raise ValueError('Period change requires at least two complete periods.')
        previous = float(period_totals.iloc[-2])
        latest = float(period_totals.iloc[-1])
        value = float(((latest - previous) / abs(previous)) * 100) if previous else np.nan
        explanation = 'Period change compares the latest period total with the immediately previous period.'

    else:
        raise ValueError('Unsupported measure type.')

    if pd.isna(value) or not np.isfinite(value):
        display_value = None
    else:
        display_value = round(float(value), 4)

    return {
        'name': name,
        'type': measure_type,
        'value': display_value,
        'explanation': explanation,
        'definition': definition,
    }
