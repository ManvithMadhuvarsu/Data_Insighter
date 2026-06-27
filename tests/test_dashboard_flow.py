"""End-to-end smoke test for the chart + dashboard pipeline through real routes.

Exercises /generate_visualization, /starter_dashboard, and /export_dashboard
against an in-memory dataset, mirroring the live UI flow (load data -> render
charts -> auto-compose a dashboard -> export). Kept to a few charts so it stays
reasonably fast.
"""

import numpy as np
import pandas as pd
import pytest

import app as app_module

TOKEN = "flow-token"


@pytest.fixture()
def sample_frame():
    rng = np.random.default_rng(11)
    n = 120
    return pd.DataFrame(
        {
            "order_date": np.repeat(pd.date_range("2023-01-01", periods=30, freq="D"), 4),
            "region": rng.choice(["North", "South", "East", "West"], size=n),
            "revenue": rng.normal(1000, 200, size=n).round(2),
            "units": rng.integers(1, 40, size=n),
        }
    )


@pytest.fixture()
def client(sample_frame, monkeypatch):
    app_module.app.config["TESTING"] = True
    record = {
        "id": "ds_flow",
        "source_name": "sales.csv",
        "owner": "analyst",
        "metadata": {"display_name": "Sales", "semantic_overrides": {}},
    }
    monkeypatch.setattr(
        app_module,
        "load_active_dataset_frame",
        lambda: (sample_frame, record, "sales.csv"),
    )
    with app_module.app.test_client() as test_client:
        with test_client.session_transaction() as state:
            state["user"] = "analyst"
            state["_csrf_token"] = TOKEN
            state["current_dataset_id"] = record["id"]
        yield test_client


@pytest.mark.parametrize(
    "columns,viz_type",
    [
        (["order_date", "revenue"], "auto"),
        (["region", "revenue"], "bar"),
        (["revenue"], "kpi"),
    ],
)
def test_generate_visualization_route_renders(client, columns, viz_type):
    response = client.post(
        "/generate_visualization",
        json={"_csrf_token": TOKEN, "columns": columns, "type": viz_type},
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["success"]
    assert payload["visualization"]["data"]


def test_starter_dashboard_then_export(client):
    composed = client.post("/starter_dashboard", json={"_csrf_token": TOKEN})
    assert composed.status_code == 200, composed.get_data(as_text=True)
    body = composed.get_json()
    dashboard_viz = body["dashboard_viz"]
    assert any(block.get("type") != "text" for block in dashboard_viz)

    exported = client.post(
        "/export_dashboard",
        json={
            "_csrf_token": TOKEN,
            "dashboard_data": {
                "dashboard_viz": dashboard_viz,
                "dashboard_state": body.get("dashboard_state", {}),
            },
        },
    )
    assert exported.status_code == 200
    assert b"<html" in exported.data.lower()
