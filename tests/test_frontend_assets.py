"""Guards on the static asset references in templates.

These are cheap text checks (no browser needed) that prevent regressions in
how the frontend loads Plotly.
"""

import re
from pathlib import Path

import pytest

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
PLOTLY_TEMPLATES = ["base.html", "analysis.html", "dashboard.html", "saving_dashboard.html"]


def _read(template_name):
    return (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")


@pytest.mark.parametrize("template_name", PLOTLY_TEMPLATES)
def test_templates_do_not_use_frozen_plotly_latest(template_name):
    # cdn.plot.ly/plotly-latest.min.js is frozen at v1.58.5 (2021) and cannot
    # render the Plotly 5.x figure JSON the backend emits.
    assert "plotly-latest" not in _read(template_name), (
        f"{template_name} references plotly-latest, which is frozen at v1.58.5 "
        "and breaks chart rendering. Pin an explicit 2.x version instead."
    )


@pytest.mark.parametrize("template_name", PLOTLY_TEMPLATES)
def test_templates_pin_an_explicit_plotly_2x_version(template_name):
    content = _read(template_name)
    assert re.search(r"plotly-2\.\d+\.\d+(\.min)?\.js", content), (
        f"{template_name} should load an explicit pinned Plotly 2.x build "
        "(e.g. plotly-2.26.0.min.js) to match the plotly.py backend."
    )
