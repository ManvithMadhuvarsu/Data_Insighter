import os
import time

import app as app_module
import workspace_store


def _configure_workspace_dirs(tmp_path, monkeypatch):
    base_dir = tmp_path / 'workspace_data'
    monkeypatch.setattr(workspace_store, 'BASE_DIR', str(base_dir))
    monkeypatch.setattr(workspace_store, 'DATASETS_DIR', str(base_dir / 'datasets'))
    monkeypatch.setattr(workspace_store, 'DASHBOARDS_DIR', str(base_dir / 'dashboards'))
    monkeypatch.setattr(workspace_store, 'RELATIONSHIPS_DIR', str(base_dir / 'relationships'))
    monkeypatch.setattr(workspace_store, 'MEASURES_DIR', str(base_dir / 'measures'))
    monkeypatch.setattr(workspace_store, 'AUDIT_DIR', str(base_dir / 'audit'))
    workspace_store.ensure_workspace_dirs()


def _set_session(client, dataset_id, filepath, token='test-token'):
    with client.session_transaction() as session_state:
        session_state['_csrf_token'] = token
        session_state['user'] = 'analyst'
        session_state['current_dataset_id'] = dataset_id
        session_state['current_filepath'] = filepath
    return token


def test_workspace_catalog_and_refresh_detect_schema_changes(tmp_path, monkeypatch):
    _configure_workspace_dirs(tmp_path, monkeypatch)
    app_module.app.config['TESTING'] = True
    original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path / 'uploads')
    os.makedirs(app_module.app.config['UPLOAD_FOLDER'], exist_ok=True)

    source = tmp_path / 'orders.csv'
    source.write_text('region,revenue\nNorth,100\nSouth,200\n', encoding='utf-8')

    dataset_record = workspace_store.create_dataset_record(
        'analyst',
        source_name='orders.csv',
        stored_path=str(source),
        source_type='upload',
        row_count=2,
        column_count=2,
        metadata={
            'display_name': 'orders.csv',
            'columns': ['region', 'revenue'],
            'lineage_steps': [],
            'pipeline_steps': [],
            'schema_snapshot': {
                'columns': ['region', 'revenue'],
                'dtypes': {'region': 'object', 'revenue': 'int64'},
            },
            'last_refreshed_at': '2026-01-01T00:00:00Z',
        },
    )

    time.sleep(1)
    source.write_text('region,revenue,margin\nNorth,100,25\nSouth,200,50\n', encoding='utf-8')

    try:
        with app_module.app.test_client() as client:
            token = _set_session(client, dataset_record['id'], str(source))

            catalog_response = client.get('/workspace_catalog')
            catalog_payload = catalog_response.get_json()
            assert catalog_response.status_code == 200
            assert catalog_payload['datasets'][0]['freshness']['status'] == 'stale_source'

            refresh_response = client.post(f"/datasets/{dataset_record['id']}/refresh", json={'_csrf_token': token})
            refresh_payload = refresh_response.get_json()
            assert refresh_response.status_code == 200
            assert refresh_payload['schema_changes']['added_columns'] == ['margin']
            assert refresh_payload['dataset']['column_count'] == 3
    finally:
        app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder
