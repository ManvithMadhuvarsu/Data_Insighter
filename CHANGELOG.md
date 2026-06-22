# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Meaningful aggregation** for charts so high-cardinality or raw-row data no
  longer renders as noise: line charts aggregate duplicate x values into one
  point per period (sorted chronologically); bar charts sort by value and cap to
  the top 25; pie/donut charts cap to the top 12 slices and bucket the remainder
  into an exact-total "Other".
- **Auto chart recommendation** (`chart_recommender.py`): a semantic-type-driven
  "auto visual" that picks the best chart for the selected columns, mirroring the
  Power BI / Fabric "Suggest a chart" experience. Available as the
  `Auto (recommended)` option in the analysis builder and as `viz_type='auto'`.
- **New chart types**: area, donut, treemap, funnel, and bubble charts, in both
  the rendering engine and the chart-type picker.
- **Tests**: first coverage for `visualization_generator.py` plus a dedicated
  suite for the chart recommender.
- Project hygiene: MIT `LICENSE`, `CONTRIBUTING.md`, `pyproject.toml`, `.flake8`,
  and a GitHub Actions CI workflow (lint + tests on Python 3.10/3.11).

### Fixed
- Single-column **box plots** crashed because the outlier mask used
  `a > x | b < y` without parentheses (operator precedence). Now wrapped.
- **Datetime-axis charts** (line/area/time-series) failed to serialize with
  "Object of type datetime is not JSON serializable"; `NumpyEncoder` now handles
  `datetime`, `date`, and `numpy.datetime64`.
- Selecting the **same column twice** crashed chart generation; duplicate
  selections are now collapsed.
- **scatter trendlines** crashed on fresh installs because `statsmodels` (needed
  by Plotly's OLS trendline) was missing from `requirements.txt`; now declared.
