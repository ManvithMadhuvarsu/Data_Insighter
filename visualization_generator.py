import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import json
import os
from datetime import date, datetime
import numpy as np
from scipy import stats
import warnings
from sklearn.linear_model import LinearRegression
from dataset_runtime import load_prepared_dataframe, prepare_dataframe

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.complex_, np.complex64, np.complex128)):
            return {'real': o.real, 'imag': o.imag}
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        elif isinstance(o, (np.bool_)):
            return bool(o)
        elif isinstance(o, (np.void)):
            return None
        elif o is pd.NaT:
            # NaT subclasses datetime, so it must be caught before the
            # datetime branches; strftime on NaT raises.
            return None
        elif isinstance(o, pd.Timestamp):
            return None if pd.isna(o) else o.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, datetime):
            return None if pd.isna(o) else o.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, date):
            return o.strftime('%Y-%m-%d')
        elif isinstance(o, np.datetime64):
            timestamp = pd.Timestamp(o)
            return None if pd.isna(timestamp) else timestamp.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, pd.Series):
            return o.tolist()
        elif isinstance(o, pd.DataFrame):
            return o.to_dict('records')
        return super().default(o)

class VisualizationGenerator:
    def __init__(self, filepath: str = None, dataframe: pd.DataFrame | None = None):
        self.filepath = filepath
        self.df = None  # Initialize df to None first

        if dataframe is not None:
            self.df = prepare_dataframe(dataframe)
        elif filepath:
            try:
                self.df = load_prepared_dataframe(filepath)

                if self.df is None or self.df.empty:
                    raise ValueError("Could not read the file or file is empty")
            except Exception as e:
                self.df = None  # Ensure df is None if there's an error
                raise ValueError(f"Error loading data: {str(e)}") from e
        else:
            self.df = None
    
    def generate_visualizations(self, columns: List[str]) -> Dict[str, Any]:
        """Generate various visualizations for selected columns."""
        if self.df is None:
            raise ValueError("No data loaded")
            
        visualizations = {
            'distributions': self._generate_distributions(columns),
            'correlations': self._generate_correlation_plot(columns),
            'time_series': self._generate_time_series(columns),
            'categorical': self._generate_categorical_plots(columns)
        }
        return visualizations
    
    def _generate_distributions(self, columns: List[str]) -> Dict[str, Any]:
        """Generate distribution plots for numeric columns."""
        distributions = {}
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                fig = px.histogram(self.df, x=col, title=f'Distribution of {col}')
                distributions[col] = fig.to_json()
        
        return distributions
    
    def _generate_correlation_plot(self, columns: List[str]) -> Dict[str, Any]:
        """Generate correlation matrix plot for numeric columns."""
        numeric_cols = [col for col in columns 
                       if pd.api.types.is_numeric_dtype(self.df[col])]
        
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                          labels=dict(color="Correlation"),
                          title="Correlation Matrix")
            return fig.to_json()
        return None
    
    def _generate_time_series(self, columns: List[str]) -> Dict[str, Any]:
        """Generate time series plots if date column is present."""
        date_cols = [col for col in self.df.columns 
                    if pd.api.types.is_datetime64_any_dtype(self.df[col])]
        
        time_series = {}
        if date_cols and columns:
            for date_col in date_cols:
                for col in columns:
                    if col != date_col and pd.api.types.is_numeric_dtype(self.df[col]):
                        fig = px.line(self.df, x=date_col, y=col,
                                    title=f'{col} over time')
                        time_series[f"{date_col}_{col}"] = fig.to_json()
        
        return time_series
    
    def _generate_categorical_plots(self, columns: List[str]) -> Dict[str, Any]:
        """Generate plots for categorical columns."""
        categorical = {}
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                value_counts = self.df[col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                            title=f'Distribution of {col}')
                categorical[col] = fig.to_json()
        
        return categorical
    
    def generate_dashboard(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard based on configuration."""
        if self.df is None:
            raise ValueError("No data loaded")
            
        dashboard_data = {
            'layout': config.get('layout', []),
            'visualizations': {}
        }
        
        for viz_config in config.get('visualizations', []):
            viz_type = viz_config.get('type')
            columns = viz_config.get('columns', [])
            
            if viz_type == 'distribution':
                dashboard_data['visualizations'][viz_config['id']] = \
                    self._generate_distributions(columns)
            elif viz_type == 'correlation':
                dashboard_data['visualizations'][viz_config['id']] = \
                    self._generate_correlation_plot(columns)
            elif viz_type == 'time_series':
                dashboard_data['visualizations'][viz_config['id']] = \
                    self._generate_time_series(columns)
            elif viz_type == 'categorical':
                dashboard_data['visualizations'][viz_config['id']] = \
                    self._generate_categorical_plots(columns)
        
        return dashboard_data
    
    def export_visualization(self, viz_type: str, viz_data: Dict[str, Any], output_dir: str) -> str:
        """Export visualization to file."""
        if viz_type == 'png':
            # Convert plotly JSON to figure and save as PNG
            fig = go.Figure(viz_data)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            )
            fig.write_image(output_file)
            return output_file
        
        raise ValueError(f"Unsupported export type: {viz_type}")
    
    def _sample_data(self, percentage: int) -> pd.DataFrame:
        """Sample the dataframe based on percentage."""
        if percentage >= 100:
            return self.df
        
        sample_size = max(1, int(len(self.df) * (percentage / 100)))
        return self.df.sample(n=sample_size, random_state=42)

    def _prepare_subset(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df[columns].copy()
        df = df.dropna(how='all')

        for col in columns:
            if df[col].dtype == 'object':
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().mean() >= 0.8:
                    df[col] = numeric_series
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                if datetime_series.notna().mean() >= 0.8:
                    df[col] = datetime_series

        return df.dropna(how='all')

    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any] | None) -> pd.DataFrame:
        if not filters:
            return df

        working = df.copy()
        column = filters.get('column')
        value = filters.get('value')
        if column and column in working.columns and value not in (None, '', '__all__'):
            working = working[working[column].astype(str) == str(value)]

        date_column = filters.get('date_column')
        if date_column and date_column in working.columns:
            dates = pd.to_datetime(working[date_column], errors='coerce')
            start_date = filters.get('start_date')
            end_date = filters.get('end_date')
            if start_date:
                working = working[dates >= pd.to_datetime(start_date, errors='coerce')]
                dates = pd.to_datetime(working[date_column], errors='coerce')
            if end_date:
                working = working[dates <= pd.to_datetime(end_date, errors='coerce')]

        return working

    # Readability caps so high-cardinality dimensions do not produce
    # thousands of unreadable bars / hundreds of pie slices.
    _MAX_BAR_CATEGORIES = 25
    _MAX_PIE_SLICES = 12

    def _aggregate_by_dimension(self, df, dimension, measure=None, aggfunc='sum'):
        """Aggregate ``measure`` (or a row count) grouped by ``dimension``.

        Returns ``(frame, value_col)`` sorted by the aggregate descending so
        the most important categories read first, like Power BI's default
        sort-by-value behavior.
        """
        if measure is None:
            agg = df.groupby(dimension).size().reset_index(name='count')
            value_col = 'count'
        elif aggfunc == 'count':
            agg = df.groupby(dimension)[measure].count().reset_index()
            value_col = measure
        else:
            agg = df.groupby(dimension)[measure].sum().reset_index()
            value_col = measure
        return agg.sort_values(value_col, ascending=False).reset_index(drop=True), value_col

    def _limit_categories(self, agg, value_col, top_n, other_bucket=False):
        """Trim an aggregated frame to the top ``top_n`` rows.

        When ``other_bucket`` is set the remaining rows are collapsed into a
        single "Other" row (used for part-to-whole charts so the total stays
        correct). Returns ``(frame, truncated)``.
        """
        if len(agg) <= top_n:
            return agg, False
        top = agg.head(top_n).copy()
        if other_bucket:
            dim_col = next(c for c in agg.columns if c != value_col)
            other_value = agg[value_col].iloc[top_n:].sum()
            other_row = pd.DataFrame([{dim_col: 'Other', value_col: other_value}])
            top = pd.concat([top, other_row], ignore_index=True)
        return top, True
        
    def generate_visualization(self, columns: List[str], viz_type: str, sample_percentage: int = 100, filters: Dict[str, Any] | None = None) -> Dict:
        """Generate a single visualization based on columns and type."""
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            if not columns:
                raise ValueError("No columns selected")

            # Drop duplicate selections (keeping order). Selecting the same
            # field twice would otherwise yield duplicate-named columns and a
            # confusing "'DataFrame' object has no attribute 'dtype'" crash.
            columns = list(dict.fromkeys(columns))

            # Verify all columns exist
            missing_cols = [col for col in columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {', '.join(missing_cols)}")

            # 'auto' resolves to the best chart type for the selected columns,
            # mirroring the Power BI / Fabric "Suggest a chart" experience.
            auto_reason = None
            if viz_type == 'auto':
                from chart_recommender import recommend_chart
                recommendation = recommend_chart(self.df, columns)
                viz_type = recommendation['type']
                columns = recommendation['columns']
                auto_reason = recommendation['reason']

            df = self._sample_data(sample_percentage)
            df = self._apply_filters(df, filters)
            df = self._prepare_subset(df, columns)
            if df.empty:
                raise ValueError("The selected columns do not contain enough usable values to visualize")
            
            # Enhanced statistical analysis
            def add_statistical_annotations(fig, df, x_col, y_col=None):
                """Add statistical annotations to the plot."""
                if y_col is None:
                    # Univariate statistics
                    stats_text = f"""
                    Mean: {df[x_col].mean():.2f}
                    Median: {df[x_col].median():.2f}
                    Std Dev: {df[x_col].std():.2f}
                    Skewness: {df[x_col].skew():.2f}
                    Kurtosis: {df[x_col].kurtosis():.2f}
                    """
                else:
                    # Bivariate statistics
                    correlation = df[x_col].corr(df[y_col])
                    stats_text = f"""
                    Correlation: {correlation:.2f}
                    R-squared: {correlation**2:.2f}
                    """
                
                fig.add_annotation(
                    text=stats_text,
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=1.02,
                    y=0.95,
                    bordercolor='black',
                    borderwidth=1
                )
                return fig
            
            # Generate visualization based on type
            if viz_type == 'bar':
                if len(columns) == 1:
                    # Single column: rank values by frequency, capped for readability.
                    agg_df, value_col = self._aggregate_by_dimension(df, columns[0])
                    agg_df, truncated = self._limit_categories(agg_df, value_col, self._MAX_BAR_CATEGORIES)
                    title = f'Distribution of {columns[0]}'
                    if truncated:
                        title += f' (top {self._MAX_BAR_CATEGORIES})'
                    fig = px.bar(agg_df, x=columns[0], y=value_col, title=title,
                                 labels={value_col: 'Count'})
                else:
                    # First column groups; second is summed (numeric) or counted.
                    if pd.api.types.is_numeric_dtype(df[columns[1]]):
                        agg_df, value_col = self._aggregate_by_dimension(df, columns[0], columns[1], 'sum')
                        title = f'Sum of {columns[1]} by {columns[0]}'
                    else:
                        agg_df, value_col = self._aggregate_by_dimension(df, columns[0], columns[1], 'count')
                        title = f'Count of {columns[1]} by {columns[0]}'
                    agg_df, truncated = self._limit_categories(agg_df, value_col, self._MAX_BAR_CATEGORIES)
                    if truncated:
                        title += f' (top {self._MAX_BAR_CATEGORIES})'
                    fig = px.bar(agg_df, x=columns[0], y=value_col, title=title)
            
            elif viz_type == 'line':
                if len(columns) < 2:
                    raise ValueError("Line chart requires at least 2 columns")
                y_cols = [col for col in columns[1:] if pd.api.types.is_numeric_dtype(df[col])]
                if not y_cols:
                    raise ValueError("Line chart needs at least one numeric column to plot over the x-axis")
                # Drop rows with no x value; an undated point is meaningless on
                # a trend axis (and NaT cannot be serialized).
                df = df.dropna(subset=[columns[0]])
                # Aggregate duplicate x values so the trend is a clean series
                # instead of a zig-zag connecting every raw row.
                if df[columns[0]].duplicated().any():
                    df = df.groupby(columns[0], as_index=False)[y_cols].sum()
                df = df.sort_values(by=columns[0])
                fig = px.line(df, x=columns[0], y=y_cols,
                              title=f'{", ".join(y_cols)} over {columns[0]}')
            
            elif viz_type == 'scatter':
                if len(columns) < 2:
                    raise ValueError("Scatter plot requires at least 2 columns")
                if not all(pd.api.types.is_numeric_dtype(df[col]) for col in columns[:2]):
                    raise ValueError("Both columns must be numeric for scatter plot")
                
                fig = px.scatter(
                    df, 
                    x=columns[0], 
                    y=columns[1],
                    title=f'Scatter Plot of {columns[0]} vs {columns[1]}',
                    trendline='ols'
                )
                
                # Add statistical annotations
                fig = add_statistical_annotations(fig, df, columns[0], columns[1])
                
                # Add confidence intervals
                model = LinearRegression()
                model.fit(df[[columns[0]]], df[columns[1]])
                y_pred = model.predict(df[[columns[0]]])
                residuals = df[columns[1]] - y_pred
                std_residuals = np.std(residuals)
                
                fig.add_trace(go.Scatter(
                    x=df[columns[0]],
                    y=y_pred + 1.96 * std_residuals,
                    mode='lines',
                    name='Upper CI',
                    line=dict(color='rgba(0,100,80,0.2)')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df[columns[0]],
                    y=y_pred - 1.96 * std_residuals,
                    mode='lines',
                    name='Lower CI',
                    line=dict(color='rgba(0,100,80,0.2)'),
                    fill='tonexty'
                ))
            
            elif viz_type == 'pie':
                if len(columns) != 2:
                    raise ValueError("Pie chart requires exactly 2 columns")
                if not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    agg_df, value_col = self._aggregate_by_dimension(df, columns[0])
                    title = f'Distribution of {columns[0]}'
                else:
                    agg_df, value_col = self._aggregate_by_dimension(df, columns[0], columns[1], 'sum')
                    title = f'Sum of {columns[1]} by {columns[0]}'
                agg_df, _ = self._limit_categories(agg_df, value_col, self._MAX_PIE_SLICES, other_bucket=True)
                fig = px.pie(agg_df, names=columns[0], values=value_col, title=title)
            
            elif viz_type == 'histogram':
                if not pd.api.types.is_numeric_dtype(df[columns[0]]):
                    raise ValueError(f"Column {columns[0]} must be numeric for histogram")
                
                # Calculate optimal number of bins using Freedman-Diaconis rule
                iqr = df[columns[0]].quantile(0.75) - df[columns[0]].quantile(0.25)
                if iqr and len(df) > 1:
                    bin_width = 2 * iqr / (len(df) ** (1/3))
                else:
                    bin_width = 0

                if bin_width and np.isfinite(bin_width) and bin_width > 0:
                    num_bins = int((df[columns[0]].max() - df[columns[0]].min()) / bin_width)
                else:
                    num_bins = min(max(int(np.sqrt(len(df))), 10), 60)
                num_bins = max(num_bins, 5)
                
                fig = px.histogram(
                    df, 
                    x=columns[0],
                    nbins=num_bins,
                    title=f'Distribution of {columns[0]}',
                    marginal='box'
                )
                
                # Add statistical annotations
                fig = add_statistical_annotations(fig, df, columns[0])
                
                # Add normal distribution overlay
                x_range = np.linspace(df[columns[0]].min(), df[columns[0]].max(), 100)
                std = df[columns[0]].std()
                if std and np.isfinite(std) and std > 0:
                    y_range = stats.norm.pdf(x_range, df[columns[0]].mean(), std)
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_range * len(df) * (df[columns[0]].max() - df[columns[0]].min()) / num_bins,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', dash='dash')
                    ))
            
            elif viz_type == 'box':
                if len(columns) == 1:
                    if not pd.api.types.is_numeric_dtype(df[columns[0]]):
                        raise ValueError(f"Column {columns[0]} must be numeric for box plot")
                    
                    fig = px.box(
                        df, 
                        y=columns[0],
                        title=f'Box Plot of {columns[0]}',
                        points='all'
                    )
                    
                    # Add statistical annotations
                    fig = add_statistical_annotations(fig, df, columns[0])
                    
                    # Add outlier annotations
                    q1 = df[columns[0]].quantile(0.25)
                    q3 = df[columns[0]].quantile(0.75)
                    iqr = q3 - q1
                    outliers = df[(df[columns[0]] > (q3 + 1.5 * iqr)) |
                                  (df[columns[0]] < (q1 - 1.5 * iqr))]
                    
                    for _idx, row in outliers.iterrows():
                        fig.add_annotation(
                            x=0,
                            y=row[columns[0]],
                            text=f'Outlier: {row[columns[0]]:.2f}',
                            showarrow=True,
                            arrowhead=1
                        )
                
                else:
                    if not pd.api.types.is_numeric_dtype(df[columns[1]]):
                        raise ValueError(f"Column {columns[1]} must be numeric for box plot")
                    
                    fig = px.box(
                        df, 
                        x=columns[0], 
                        y=columns[1],
                        title=f'Box Plot of {columns[1]} by {columns[0]}',
                        points='all'
                    )
                    
                    # Note: per-group stat annotations are intentionally omitted.
                    # They previously all rendered at the same paper coordinate
                    # and overlapped into an unreadable blob; the box glyphs and
                    # hover already expose each group's quartiles and outliers.

            elif viz_type == 'kpi':
                if not pd.api.types.is_numeric_dtype(df[columns[0]]):
                    raise ValueError(f"Column {columns[0]} must be numeric for a KPI card")
                value = float(df[columns[0]].sum())
                average = float(df[columns[0]].mean())
                fig = go.Figure(go.Indicator(
                    mode='number+delta',
                    value=value,
                    number={'valueformat': ',.2f'},
                    delta={'reference': average, 'relative': False, 'valueformat': ',.2f'},
                    title={'text': f"Total {columns[0]}<br><span style='font-size:0.7em;color:gray'>Delta vs average row value</span>"},
                ))

            elif viz_type == 'heatmap':
                if len(columns) < 2:
                    raise ValueError("Heatmap requires two columns")
                if len(columns) >= 3 and pd.api.types.is_numeric_dtype(df[columns[2]]):
                    pivot = df.pivot_table(index=columns[1], columns=columns[0], values=columns[2], aggfunc='sum', fill_value=0)
                    title = f'Sum of {columns[2]} by {columns[0]} and {columns[1]}'
                else:
                    pivot = pd.crosstab(df[columns[1]].astype(str), df[columns[0]].astype(str))
                    title = f'Count heatmap of {columns[0]} by {columns[1]}'
                fig = px.imshow(
                    pivot,
                    text_auto=True,
                    aspect='auto',
                    title=title,
                    labels={'x': columns[0], 'y': columns[1], 'color': 'Value'},
                )
            
            elif viz_type == 'area':
                if len(columns) < 2:
                    raise ValueError("Area chart requires at least 2 columns")
                y_cols = [col for col in columns[1:] if pd.api.types.is_numeric_dtype(df[col])]
                if not y_cols:
                    raise ValueError("Area chart needs at least one numeric column to plot over the x-axis")
                # Drop rows with no x value, then aggregate duplicate x so the
                # filled area reads as a clean series instead of overlapping
                # raw rows.
                df = df.dropna(subset=[columns[0]])
                if df[columns[0]].duplicated().any():
                    df = df.groupby(columns[0], as_index=False)[y_cols].sum()
                if pd.api.types.is_numeric_dtype(df[columns[0]]) or pd.api.types.is_datetime64_any_dtype(df[columns[0]]):
                    df = df.sort_values(by=columns[0])
                fig = px.area(df, x=columns[0], y=y_cols,
                              title=f'{", ".join(y_cols)} over {columns[0]}')

            elif viz_type == 'donut':
                if len(columns) != 2:
                    raise ValueError("Donut chart requires exactly 2 columns")
                if not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    agg_df, value_col = self._aggregate_by_dimension(df, columns[0])
                    title = f'Distribution of {columns[0]}'
                else:
                    agg_df, value_col = self._aggregate_by_dimension(df, columns[0], columns[1], 'sum')
                    title = f'Sum of {columns[1]} by {columns[0]}'
                agg_df, _ = self._limit_categories(agg_df, value_col, self._MAX_PIE_SLICES, other_bucket=True)
                fig = px.pie(agg_df, names=columns[0], values=value_col, hole=0.5, title=title)

            elif viz_type == 'treemap':
                if len(columns) < 2:
                    raise ValueError("Treemap requires a category column and a value column")
                if pd.api.types.is_numeric_dtype(df[columns[-1]]):
                    path_cols = columns[:-1]
                    value_col = columns[-1]
                    agg_df = df.groupby(path_cols)[value_col].sum().reset_index()
                    fig = px.treemap(agg_df, path=path_cols, values=value_col,
                                     title=f'{value_col} by {", ".join(path_cols)}')
                else:
                    agg_df = df.groupby(columns).size().reset_index(name='count')
                    fig = px.treemap(agg_df, path=columns, values='count',
                                     title=f'Record count by {", ".join(columns)}')

            elif viz_type == 'funnel':
                if len(columns) < 2:
                    raise ValueError("Funnel chart requires a stage column and a numeric value column")
                if not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    raise ValueError(f"Column {columns[1]} must be numeric for a funnel chart")
                agg_df = (df.groupby(columns[0])[columns[1]].sum()
                          .reset_index().sort_values(columns[1], ascending=False))
                fig = px.funnel(agg_df, x=columns[1], y=columns[0],
                                title=f'{columns[1]} funnel by {columns[0]}')

            elif viz_type == 'bubble':
                if len(columns) < 3:
                    raise ValueError("Bubble chart requires 3 columns: x, y, and a size measure")
                if not all(pd.api.types.is_numeric_dtype(df[col]) for col in columns[:3]):
                    raise ValueError("Bubble chart requires numeric x, y, and size columns")
                sized = df.assign(_bubble_size=df[columns[2]].abs())
                color_col = columns[3] if len(columns) > 3 else None
                fig = px.scatter(sized, x=columns[0], y=columns[1], size='_bubble_size',
                                 color=color_col, size_max=40,
                                 title=f'{columns[1]} vs {columns[0]} sized by {columns[2]}')

            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Update layout with enhanced settings
            layout_settings = {
                'width': 1000,  # Increased width for better visibility
                'height': 600,
                'template': 'plotly_white',
                'font': {'size': 12},
                'margin': {'l': 50, 'r': 200, 't': 50, 'b': 50},  # Increased right margin for annotations
                'showlegend': True,
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'hovermode': 'closest',
                'hoverlabel': {'bgcolor': 'white', 'font_size': 12}
            }
            
            fig.update_layout(**layout_settings)

            # Convert to dict and handle JSON serialization
            fig_dict = fig.to_dict()
            if auto_reason:
                # Record what 'auto' resolved to so the client can explain it.
                meta = fig_dict.setdefault('layout', {}).setdefault('meta', {})
                if isinstance(meta, dict):
                    meta['auto_recommendation'] = {'type': viz_type, 'columns': columns, 'reason': auto_reason}
            return json.loads(json.dumps(fig_dict, cls=NumpyEncoder))
            
        except Exception as e:
            import traceback
            print(f"Error generating visualization: {str(e)}")
            print(traceback.format_exc())
            raise ValueError(f"Error generating visualization: {str(e)}") from e
