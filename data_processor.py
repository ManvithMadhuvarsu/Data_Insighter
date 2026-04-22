import math
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from file_utils import read_data_file
from insight_engine import (
    anomaly_insights,
    build_executive_takeaways,
    contribution_insights,
    funnel_insights,
    retention_cohort_insights,
    seasonality_insights,
    segment_driver_insights,
    variance_explanation_insights,
)
from semantic_model import infer_dataset_semantics


class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = self._load_data()
        self.df = self._prepare_dataframe(self.df)

    def _load_data(self) -> pd.DataFrame:
        """Load data from file with enhanced format support and error handling."""
        try:
            return read_data_file(self.filepath)
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim names, drop empty rows/cols, and infer basic semantic types."""
        prepared = df.copy()
        prepared.columns = [str(column).strip() for column in prepared.columns]
        prepared = prepared.dropna(axis=1, how='all').dropna(how='all').reset_index(drop=True)

        for column in prepared.columns:
            series = prepared[column]

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna().astype(str).str.strip()
                if non_null.empty:
                    continue

                numeric_series = pd.to_numeric(non_null.str.replace(",", "", regex=False), errors='coerce')
                numeric_ratio = numeric_series.notna().mean()
                if numeric_ratio >= 0.8:
                    prepared[column] = pd.to_numeric(
                        series.astype(str).str.replace(",", "", regex=False),
                        errors='coerce'
                    )
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    datetime_series = pd.to_datetime(series, errors='coerce')
                datetime_ratio = datetime_series.notna().mean()
                if datetime_ratio >= 0.8:
                    prepared[column] = datetime_series

        return prepared

    def _serialize_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            if math.isfinite(float(value)):
                return round(float(value), 4)
            return None
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        return str(value)

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        return round((numerator / denominator) * 100, 2) if denominator else 0.0

    def _numeric_columns(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def _datetime_columns(self) -> List[str]:
        return self.df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()

    def _categorical_columns(self) -> List[str]:
        excluded = set(self._numeric_columns()) | set(self._datetime_columns())
        candidates = [column for column in self.df.columns if column not in excluded]
        return candidates

    def _semantic_profiles(self) -> List[Dict[str, Any]]:
        return infer_dataset_semantics(self.df)

    def _semantic_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {
            'measure': [],
            'dimension': [],
            'datetime': [],
            'identifier': [],
        }
        for profile in self._semantic_profiles():
            groups.setdefault(profile['semantic_role'], []).append(profile['name'])
        return groups

    def _column_profiles(self) -> List[Dict[str, Any]]:
        profiles: List[Dict[str, Any]] = []

        for column in self.df.columns:
            series = self.df[column]
            non_null = series.dropna()
            profile = {
                'name': column,
                'dtype': str(series.dtype),
                'non_null_count': int(non_null.shape[0]),
                'missing_count': int(series.isna().sum()),
                'missing_pct': self._safe_ratio(series.isna().sum(), len(series)),
                'unique_count': int(non_null.nunique()),
                'sample_values': [self._serialize_value(value) for value in non_null.head(3).tolist()],
            }

            if pd.api.types.is_numeric_dtype(series):
                clean = non_null.astype(float)
                if not clean.empty:
                    q1 = clean.quantile(0.25)
                    q3 = clean.quantile(0.75)
                    iqr = q3 - q1
                    outlier_count = 0
                    if iqr > 0:
                        outlier_mask = (clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))
                        outlier_count = int(outlier_mask.sum())

                    profile.update({
                        'role': 'numeric',
                        'mean': self._serialize_value(clean.mean()),
                        'median': self._serialize_value(clean.median()),
                        'std': self._serialize_value(clean.std()),
                        'min': self._serialize_value(clean.min()),
                        'max': self._serialize_value(clean.max()),
                        'outlier_pct': self._safe_ratio(outlier_count, len(clean)),
                        'skewness': self._serialize_value(clean.skew()),
                    })
            elif pd.api.types.is_datetime64_any_dtype(series):
                profile.update({
                    'role': 'datetime',
                    'min': self._serialize_value(non_null.min() if not non_null.empty else None),
                    'max': self._serialize_value(non_null.max() if not non_null.empty else None),
                })
            else:
                top_counts = non_null.astype(str).value_counts().head(3)
                profile.update({
                    'role': 'categorical',
                    'top_values': [
                        {
                            'label': value,
                            'count': int(count),
                            'pct': self._safe_ratio(count, len(non_null)),
                        }
                        for value, count in top_counts.items()
                    ]
                })

            profiles.append(profile)

        return profiles

    def _quality_alerts(self) -> List[Dict[str, str]]:
        alerts: List[Dict[str, str]] = []
        row_count = len(self.df)
        duplicate_count = int(self.df.duplicated().sum())
        duplicate_pct = self._safe_ratio(duplicate_count, row_count)

        if duplicate_count:
            alerts.append({
                'severity': 'warning',
                'title': 'Duplicate rows detected',
                'detail': f'{duplicate_count} rows ({duplicate_pct}%) are exact duplicates and may distort aggregates.'
            })

        for profile in self._column_profiles():
            if profile['missing_pct'] >= 30:
                alerts.append({
                    'severity': 'warning',
                    'title': f"High missingness in '{profile['name']}'",
                    'detail': f"{profile['missing_pct']}% of this column is blank, so insights using it may be incomplete."
                })

            if profile.get('role') == 'numeric' and profile.get('outlier_pct', 0) >= 10:
                alerts.append({
                    'severity': 'info',
                    'title': f"Potential outliers in '{profile['name']}'",
                    'detail': f"{profile['outlier_pct']}% of values sit outside the IQR range. Review before using averages."
                })

        if not alerts:
            alerts.append({
                'severity': 'success',
                'title': 'No major data quality blockers found',
                'detail': 'The dataset looks usable for exploratory analysis with the current checks.'
            })

        return alerts[:8]

    def _correlation_insights(self) -> List[Dict[str, Any]]:
        numeric_columns = self._numeric_columns()
        if len(numeric_columns) < 2:
            return []

        corr = self.df[numeric_columns].corr(numeric_only=True)
        findings = []

        for idx, left in enumerate(numeric_columns):
            for right in numeric_columns[idx + 1:]:
                value = corr.loc[left, right]
                if pd.isna(value):
                    continue
                strength = abs(float(value))
                if strength >= 0.6:
                    findings.append({
                        'kind': 'correlation',
                        'title': f'{left} and {right} move together',
                        'detail': f'The correlation is {value:.2f}, which is strong enough to investigate a business relationship.',
                        'score': round(strength, 3),
                        'stat': f'r = {value:.2f}',
                        'recommended_chart': {
                            'title': f'{left} vs {right}',
                            'description': 'Use a scatter plot to inspect the relationship and spot clusters or non-linear behavior.',
                            'type': 'scatter',
                            'columns': [left, right],
                            'sample_percentage': 100,
                        }
                    })

        findings.sort(key=lambda item: item['score'], reverse=True)
        return findings[:3]

    def _distribution_insights(self) -> List[Dict[str, Any]]:
        findings = []
        for column in self._numeric_columns():
            series = self.df[column].dropna().astype(float)
            if len(series) < 8:
                continue

            skewness = series.skew()
            if pd.notna(skewness) and abs(float(skewness)) >= 1:
                direction = 'right-skewed' if skewness > 0 else 'left-skewed'
                findings.append({
                    'kind': 'distribution',
                    'title': f'{column} has a {direction} distribution',
                    'detail': 'The mean may be less representative than the median here, and a few values are likely stretching the range.',
                    'score': round(abs(float(skewness)), 3),
                    'stat': f'skew = {skewness:.2f}',
                    'recommended_chart': {
                        'title': f'{column} distribution',
                        'description': 'Inspect the distribution and the spread before making KPI claims from averages.',
                        'type': 'histogram',
                        'columns': [column],
                        'sample_percentage': 100,
                    }
                })

        findings.sort(key=lambda item: item['score'], reverse=True)
        return findings[:2]

    def _categorical_insights(self) -> List[Dict[str, Any]]:
        findings = []
        for column in self._categorical_columns():
            series = self.df[column].dropna().astype(str)
            if series.empty:
                continue

            unique_count = series.nunique()
            if unique_count < 2 or unique_count > 20:
                continue

            top_counts = series.value_counts()
            top_label = top_counts.index[0]
            top_count = int(top_counts.iloc[0])
            dominance = self._safe_ratio(top_count, len(series))

            if dominance >= 35:
                findings.append({
                    'kind': 'categorical',
                    'title': f"{column} is concentrated in '{top_label}'",
                    'detail': f"The top category contributes {dominance}% of observed rows, which suggests an imbalance worth explaining.",
                    'score': dominance,
                    'stat': f'{top_label}: {dominance}%',
                    'recommended_chart': {
                        'title': f'{column} composition',
                        'description': 'See whether one segment dominates the dataset or whether the distribution is balanced.',
                        'type': 'bar',
                        'columns': [column],
                        'sample_percentage': 100,
                    }
                })

        findings.sort(key=lambda item: item['score'], reverse=True)
        return findings[:2]

    def _time_insights(self) -> List[Dict[str, Any]]:
        datetime_columns = self._datetime_columns()
        numeric_columns = self._numeric_columns()
        findings = []

        if not datetime_columns or not numeric_columns:
            return findings

        for date_column in datetime_columns[:2]:
            working = self.df[[date_column] + numeric_columns].dropna(subset=[date_column]).copy()
            if working.empty:
                continue

            working = working.sort_values(date_column)
            for numeric_column in numeric_columns[:5]:
                series = working[[date_column, numeric_column]].dropna()
                if len(series) < 8:
                    continue

                trend_delta = float(series[numeric_column].iloc[-1] - series[numeric_column].iloc[0])
                trend_pct = self._safe_ratio(trend_delta, abs(series[numeric_column].iloc[0])) if series[numeric_column].iloc[0] != 0 else 0.0
                findings.append({
                    'kind': 'trend',
                    'title': f'{numeric_column} changes over time',
                    'detail': f'From the first to last recorded period, the metric moves by {round(trend_delta, 2)}.',
                    'score': abs(trend_delta),
                    'stat': f'change = {round(trend_delta, 2)} ({trend_pct}%)',
                    'recommended_chart': {
                        'title': f'{numeric_column} over time',
                        'description': 'Use a time-series view to inspect trend shifts, spikes, and seasonality.',
                        'type': 'line',
                        'columns': [date_column, numeric_column],
                        'sample_percentage': 100,
                    }
                })

        findings.sort(key=lambda item: item['score'], reverse=True)
        return findings[:2]

    def _recommended_visualizations(self) -> List[Dict[str, Any]]:
        recommendations: List[Dict[str, Any]] = []
        seen = set()

        for collection in (
            self._correlation_insights(),
            self._time_insights(),
            self._categorical_insights(),
            self._distribution_insights(),
        ):
            for insight in collection:
                chart = insight.get('recommended_chart')
                if not chart:
                    continue
                key = (chart['type'], tuple(chart['columns']))
                if key in seen:
                    continue
                seen.add(key)
                recommendations.append(chart)

        if not recommendations:
            numeric_columns = self._numeric_columns()
            categorical_columns = self._categorical_columns()
            if numeric_columns:
                recommendations.append({
                    'title': f'{numeric_columns[0]} distribution',
                    'description': 'Start by understanding the range, skew, and extreme values of a key metric.',
                    'type': 'histogram',
                    'columns': [numeric_columns[0]],
                    'sample_percentage': 100,
                })
            if categorical_columns:
                recommendations.append({
                    'title': f'{categorical_columns[0]} breakdown',
                    'description': 'Use this to understand how the dataset is distributed across categories.',
                    'type': 'bar',
                    'columns': [categorical_columns[0]],
                    'sample_percentage': 100,
                })

        numeric_columns = self._numeric_columns()
        categorical_columns = self._categorical_columns()
        if numeric_columns and not any(chart['type'] == 'kpi' for chart in recommendations):
            recommendations.insert(0, {
                'title': f'{numeric_columns[0]} KPI',
                'description': 'Start with a headline KPI card so the dashboard has a clear business metric.',
                'type': 'kpi',
                'columns': [numeric_columns[0]],
                'sample_percentage': 100,
            })
        if len(categorical_columns) >= 2 and not any(chart['type'] == 'heatmap' for chart in recommendations):
            recommendations.append({
                'title': f'{categorical_columns[0]} by {categorical_columns[1]} heatmap',
                'description': 'Use a heatmap to find concentration patterns across two dimensions.',
                'type': 'heatmap',
                'columns': categorical_columns[:2],
                'sample_percentage': 100,
            })

        return recommendations[:6]

    def _advanced_insights(self) -> List[Dict[str, Any]]:
        semantics = self._semantic_groups()
        measures = semantics.get('measure', [])
        dimensions = semantics.get('dimension', [])
        datetimes = semantics.get('datetime', [])
        identifiers = semantics.get('identifier', [])

        findings = (
            anomaly_insights(self.df, measures)
            + segment_driver_insights(self.df, dimensions, measures)
            + contribution_insights(self.df, dimensions, measures)
            + seasonality_insights(self.df, datetimes, measures)
            + variance_explanation_insights(self.df, datetimes, dimensions, measures)
            + funnel_insights(self.df, measures)
            + retention_cohort_insights(self.df, datetimes, identifiers)
        )
        findings.sort(key=lambda item: item['score'], reverse=True)
        return findings[:6]

    def get_data_info(self) -> Dict[str, Any]:
        """Get a concise summary used by the upload flow."""
        return {
            'columns': self.df.columns.tolist(),
            'num_rows': len(self.df),
            'num_columns': len(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'preview': self.df.head(5).to_dict(orient='records'),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self._numeric_columns(),
            'categorical_columns': self._categorical_columns(),
            'datetime_columns': self._datetime_columns(),
        }

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Generate a real, dataset-driven analysis summary for the insights UI."""
        row_count = len(self.df)
        duplicate_count = int(self.df.duplicated().sum())
        numeric_columns = self._numeric_columns()
        categorical_columns = self._categorical_columns()
        datetime_columns = self._datetime_columns()

        completeness = 100 - self._safe_ratio(self.df.isna().sum().sum(), max(row_count * max(len(self.df.columns), 1), 1))

        foundational_insights = (
            self._correlation_insights()
            + self._time_insights()
            + self._categorical_insights()
            + self._distribution_insights()
        )
        advanced_insights = self._advanced_insights()
        key_insights = sorted(foundational_insights + advanced_insights, key=lambda item: item['score'], reverse=True)[:8]
        semantic_profiles = self._semantic_profiles()
        quality_alerts = self._quality_alerts()

        return {
            'dataset_overview': {
                'rows': row_count,
                'columns': int(len(self.df.columns)),
                'numeric_columns': int(len(numeric_columns)),
                'categorical_columns': int(len(categorical_columns)),
                'datetime_columns': int(len(datetime_columns)),
                'completeness_pct': round(completeness, 2),
                'duplicate_rows': duplicate_count,
                'duplicate_pct': self._safe_ratio(duplicate_count, row_count),
            },
            'quality_alerts': quality_alerts,
            'key_insights': key_insights,
            'recommended_visualizations': self._recommended_visualizations(),
            'column_profiles': self._column_profiles(),
            'semantic_profiles': semantic_profiles,
            'executive_takeaways': build_executive_takeaways(key_insights, quality_alerts),
        }
