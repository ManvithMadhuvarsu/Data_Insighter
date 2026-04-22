from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def apply_transform(df: pd.DataFrame, operation: str, options: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    working = df.copy()

    if operation == 'drop_duplicates':
        before = len(working)
        working = working.drop_duplicates().reset_index(drop=True)
        removed = before - len(working)
        return working, f'Removed {removed} duplicate rows.'

    if operation == 'trim_text':
        for column in working.select_dtypes(include=['object', 'string']).columns:
            working[column] = working[column].apply(lambda value: value.strip() if isinstance(value, str) else value)
        return working, 'Trimmed leading and trailing whitespace from text columns.'

    if operation == 'fill_numeric_median':
        numeric_columns = working.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_columns:
            median = working[column].median()
            if pd.notna(median):
                working[column] = working[column].fillna(median)
        return working, 'Filled missing numeric values with each column median.'

    if operation == 'drop_sparse_columns':
        threshold = float(options.get('threshold', 40))
        keep_columns = [
            column
            for column in working.columns
            if (working[column].isna().mean() * 100) <= threshold
        ]
        removed = [column for column in working.columns if column not in keep_columns]
        working = working[keep_columns].copy()
        return working, f"Removed {len(removed)} sparse columns above the {threshold}% missing threshold."

    if operation == 'keep_columns':
        columns = options.get('columns') or []
        missing = [column for column in columns if column not in working.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")
        return working[columns].copy(), f'Kept {len(columns)} selected columns.'

    if operation == 'create_date_parts':
        column = options.get('column')
        if not column or column not in working.columns:
            raise ValueError('Choose a valid date column first.')

        date_series = pd.to_datetime(working[column], errors='coerce')
        if date_series.notna().sum() == 0:
            raise ValueError('The selected column could not be interpreted as dates.')

        prefix = options.get('prefix') or column
        working[f'{prefix}_year'] = date_series.dt.year
        working[f'{prefix}_quarter'] = date_series.dt.quarter
        working[f'{prefix}_month'] = date_series.dt.month
        working[f'{prefix}_month_name'] = date_series.dt.month_name()
        return working, f'Created year, quarter, and month fields from {column}.'

    if operation == 'create_calculated_column':
        new_column = (options.get('new_column') or '').strip()
        left = options.get('left')
        operator = options.get('operator')
        right_mode = options.get('right_mode', 'column')
        right_value = options.get('right_value')

        if not new_column:
            raise ValueError('Enter a name for the calculated column.')
        if not left or left not in working.columns:
            raise ValueError('Choose a valid left-side column.')
        if operator not in {'+', '-', '*', '/'}:
            raise ValueError('Choose a supported arithmetic operator.')

        left_series = pd.to_numeric(working[left], errors='coerce')
        if right_mode == 'column':
            if not right_value or right_value not in working.columns:
                raise ValueError('Choose a valid right-side column.')
            right_series = pd.to_numeric(working[right_value], errors='coerce')
        else:
            try:
                constant = float(right_value)
            except (TypeError, ValueError):
                raise ValueError('Enter a numeric constant for the right side.')
            right_series = pd.Series(constant, index=working.index)

        if operator == '+':
            working[new_column] = left_series + right_series
        elif operator == '-':
            working[new_column] = left_series - right_series
        elif operator == '*':
            working[new_column] = left_series * right_series
        elif operator == '/':
            safe_right = right_series.replace({0: np.nan})
            working[new_column] = left_series / safe_right

        return working, f"Created calculated field '{new_column}'."

    raise ValueError('Unsupported transform operation.')

