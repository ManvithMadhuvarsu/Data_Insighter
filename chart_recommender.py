"""Automatic chart-type recommendation.

This module is the lightweight "auto visual" brain behind Data Insighter — the
same idea Power BI and Microsoft Fabric expose as *Suggest a chart* / *Auto*.
Given a dataframe and the columns a user selected, it picks the most
informative chart type and orders the columns so the renderer draws a sensible
axis mapping.

The logic reuses :func:`semantic_model.infer_semantic_type`, so it benefits
from the same role/cardinality inference the rest of the app already trusts
(measures vs. dimensions vs. datetime vs. identifiers) instead of looking only
at raw pandas dtypes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from semantic_model import infer_semantic_type


# Roles that behave like a grouping axis when charting.
_GROUPING_ROLES = {"dimension", "identifier"}


def _role(profile: Dict[str, Any]) -> str:
    return profile.get("semantic_role", "dimension")


def _profiles(df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
    return [infer_semantic_type(column, df[column]) for column in columns]


def _first(profiles: List[Dict[str, Any]], role: str) -> Dict[str, Any] | None:
    for profile in profiles:
        if _role(profile) == role:
            return profile
    return None


def _first_grouping(profiles: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    # Prefer lower-cardinality dimensions; they read better as an axis.
    grouping = [p for p in profiles if _role(p) in _GROUPING_ROLES]
    grouping.sort(key=lambda p: 0 if p.get("cardinality_band") in {"low", "medium"} else 1)
    return grouping[0] if grouping else None


def recommend_chart(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Recommend a chart type and column order for ``columns``.

    Returns a dict with ``type`` (a viz type understood by
    :class:`VisualizationGenerator`), ``columns`` (reordered for that type),
    and ``reason`` (a short human explanation shown in the UI).
    """
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ValueError("Select at least one column to recommend a chart")

    profiles = _profiles(df, columns)
    by_name = {p["name"]: p for p in profiles}

    measures = [p for p in profiles if _role(p) == "measure"]
    datetimes = [p for p in profiles if _role(p) == "datetime"]
    groupings = [p for p in profiles if _role(p) in _GROUPING_ROLES]

    # ---- Single column -------------------------------------------------
    if len(columns) == 1:
        only = profiles[0]
        role = _role(only)
        if role == "measure":
            return {
                "type": "histogram",
                "columns": [only["name"]],
                "reason": f"'{only['name']}' is a numeric measure, so a histogram shows its distribution and spread.",
            }
        if role == "datetime":
            return {
                "type": "bar",
                "columns": [only["name"]],
                "reason": f"'{only['name']}' is a date field; a bar of record counts per period shows volume over time.",
            }
        return {
            "type": "bar",
            "columns": [only["name"]],
            "reason": f"'{only['name']}' is a category, so a bar chart compares how records split across its values.",
        }

    # ---- Two columns ---------------------------------------------------
    if len(columns) == 2:
        if datetimes and measures:
            return {
                "type": "line",
                "columns": [datetimes[0]["name"], measures[0]["name"]],
                "reason": f"A date and a measure are selected, so a line chart shows how '{measures[0]['name']}' trends over time.",
            }
        if len(measures) == 2:
            return {
                "type": "scatter",
                "columns": [measures[0]["name"], measures[1]["name"]],
                "reason": "Two measures are selected, so a scatter plot reveals their correlation and clusters.",
            }
        if groupings and measures:
            return {
                "type": "bar",
                "columns": [groupings[0]["name"], measures[0]["name"]],
                "reason": f"A category and a measure are selected, so a bar chart compares '{measures[0]['name']}' across '{groupings[0]['name']}'.",
            }
        if len(groupings) == 2:
            return {
                "type": "heatmap",
                "columns": [groupings[0]["name"], groupings[1]["name"]],
                "reason": "Two categories are selected, so a heatmap surfaces where their combinations concentrate.",
            }
        if datetimes and groupings:
            return {
                "type": "bar",
                "columns": [datetimes[0]["name"], groupings[0]["name"]],
                "reason": "A date and a category are selected, so a bar chart compares category counts across periods.",
            }
        # Fallback: first column as axis.
        return {
            "type": "bar",
            "columns": columns,
            "reason": "A bar chart is a safe default comparison for the selected fields.",
        }

    # ---- Three or more columns ----------------------------------------
    if datetimes and measures:
        return {
            "type": "line",
            "columns": [datetimes[0]["name"]] + [m["name"] for m in measures],
            "reason": "A date plus measures are selected, so a multi-series line chart compares trends over time.",
        }
    if len(groupings) >= 2 and measures:
        return {
            "type": "heatmap",
            "columns": [groupings[0]["name"], groupings[1]["name"], measures[0]["name"]],
            "reason": f"Two categories and a measure let a heatmap show '{measures[0]['name']}' intensity across both dimensions.",
        }
    if len(measures) >= 2:
        return {
            "type": "scatter",
            "columns": [measures[0]["name"], measures[1]["name"]],
            "reason": "Multiple measures are selected, so a scatter plot compares the two strongest numeric fields.",
        }
    if groupings and measures:
        return {
            "type": "bar",
            "columns": [groupings[0]["name"], measures[0]["name"]],
            "reason": f"A bar chart compares '{measures[0]['name']}' across '{groupings[0]['name']}'.",
        }

    return {
        "type": "bar",
        "columns": columns[:2],
        "reason": "A bar chart is a safe default comparison for the selected fields.",
    }
