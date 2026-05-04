# Data Insighter

Data Insighter is a Flask-based data analysis workspace for uploading CSV/JSON datasets, profiling them, generating Plotly visualizations, creating saved dashboards, and exporting insight summaries.

## What the app currently does

- User registration and login with hashed passwords and CSRF protection
- Upload CSV and JSON files
- Load bundled sample datasets
- Generate dataset summaries, semantic profiles, quality alerts, and executive takeaways
- Apply lightweight transforms such as deduplication, text trimming, missing-value filling, date-part extraction, and calculated columns
- Suggest relationships across uploaded datasets and create joined datasets
- Define reusable business measures
- Build dashboard layouts from analysis charts and export dashboards as interactive HTML or PNG
- Export executive reports as HTML or Markdown

## Current architecture

Backend modules:

- `app.py` - Flask routes, auth, session flow, export flow
- `data_processor.py` - dataset profiling and insight summary orchestration
- `insight_engine.py` - anomaly, contribution, funnel, cohort, seasonality, and variance insights
- `semantic_model.py` - semantic role inference for columns
- `data_model_service.py` - key profiling, relationship suggestions, and joins
- `measure_service.py` - reusable measure calculations
- `transform_service.py` - dataset transformation operations
- `visualization_generator.py` - Plotly chart generation and export
- `workspace_store.py` - JSON-backed persistence for datasets, dashboards, relationships, and measures
- `report_service.py` - executive summary report generation
- `file_utils.py` - CSV/JSON loading with encoding handling

Frontend surface:

- `templates/` - Jinja views for home, upload, analysis, dashboard, auth, and export HTML
- `static/css/` - shared, upload, and analysis styling
- `static/js/upload.js` - upload and sample-dataset client flow

## Data storage model

- Uploaded files are stored under `uploads/`
- Sample files live under `sample_datasets/`
- Workspace records are stored as JSON files under `workspace_data/`
- The app uses Flask session cookies plus server-side workspace files
- If `SECRET_KEY` is not set, the app creates a stable local fallback key in `.local_secret_key`

## Supported file types

- `.csv`
- `.json`

## Local development

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Optionally create a `.env` file with:

```env
SECRET_KEY=your_secret_key_here
```

4. Start the app with:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000`.

PowerShell helpers:

- `scripts/start_local_server.ps1`
- `scripts/tail_local_logs.ps1`

## Tests

Run the current automated tests with:

```bash
python -m pytest -q
```

The current suite covers:

- measure calculations
- relationship/key profiling
- advanced insight helpers
- report HTML escaping
- auth edge cases around duplicate emails and email login

## Notes

- Dashboard browser state is scoped by user and active dataset.
- Exported report and dashboard HTML is escaped to avoid embedding executable markup from uploaded data.
- Workspace persistence is currently file-backed, not database-backed.
