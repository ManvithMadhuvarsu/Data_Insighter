from typing import Any, Dict, List

import numpy as np
import pandas as pd


GEOGRAPHY_HINTS = {'country', 'state', 'city', 'region', 'postal', 'zip', 'latitude', 'longitude', 'lat', 'lon'}
IDENTIFIER_HINTS = {'id', 'uuid', 'key', 'code', 'number', 'no'}
CURRENCY_HINTS = {'revenue', 'sales', 'price', 'cost', 'profit', 'income', 'amount', 'spend', 'expense'}
PERCENT_HINTS = {'percent', 'percentage', 'pct', 'rate', 'ratio', 'share'}
DATE_HINTS = {'date', 'time', 'month', 'year', 'day', 'week', 'quarter'}


def _contains_hint(column_name: str, hints: set[str]) -> bool:
    lowered = column_name.lower()
    return any(hint in lowered for hint in hints)


def infer_semantic_type(column: str, series: pd.Series) -> Dict[str, Any]:
    non_null = series.dropna()
    unique_count = int(non_null.nunique()) if not non_null.empty else 0
    uniqueness_ratio = unique_count / max(len(non_null), 1)
    lowered = column.lower()

    semantic_role = 'dimension'
    subtype = 'category'
    rationale: List[str] = []

    if pd.api.types.is_datetime64_any_dtype(series) or _contains_hint(lowered, DATE_HINTS):
        semantic_role = 'datetime'
        subtype = 'date'
        rationale.append('The field is parsed as a date or its name strongly implies time-based values.')
    elif pd.api.types.is_numeric_dtype(series):
        semantic_role = 'measure'
        subtype = 'numeric'
        rationale.append('The field behaves like a numeric measure suitable for aggregation.')

        if _contains_hint(lowered, PERCENT_HINTS):
            subtype = 'percentage'
            rationale.append('Its name suggests a percentage or rate.')
        elif _contains_hint(lowered, CURRENCY_HINTS):
            subtype = 'currency'
            rationale.append('Its name suggests a financial amount.')
        elif (_contains_hint(lowered, IDENTIFIER_HINTS) and uniqueness_ratio > 0.9) or column.lower().endswith('_id'):
            semantic_role = 'identifier'
            subtype = 'identifier'
            rationale.append('Although numeric, the values are highly unique and the name looks like an identifier.')
    else:
        if (_contains_hint(lowered, IDENTIFIER_HINTS) and uniqueness_ratio > 0.9) or lowered.endswith('_id'):
            semantic_role = 'identifier'
            subtype = 'identifier'
            rationale.append('The field name and uniqueness pattern suggest row-level identifiers.')
        elif _contains_hint(lowered, GEOGRAPHY_HINTS):
            semantic_role = 'dimension'
            subtype = 'geography'
            rationale.append('The field name suggests a geographic dimension.')
        else:
            semantic_role = 'dimension'
            subtype = 'category'
            rationale.append('The field is best treated as a grouping dimension or segment.')

    if semantic_role == 'dimension' and unique_count > 0 and unique_count <= 12:
        rationale.append('Its cardinality is low enough to work well in grouped comparisons and filters.')
    elif semantic_role == 'dimension' and unique_count > 50:
        rationale.append('Its high cardinality may require ranking or filtering before charting.')

    return {
        'name': column,
        'semantic_role': semantic_role,
        'subtype': subtype,
        'unique_count': unique_count,
        'uniqueness_ratio': round(uniqueness_ratio, 4),
        'rationale': rationale,
    }


def infer_dataset_semantics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [infer_semantic_type(column, df[column]) for column in df.columns]

