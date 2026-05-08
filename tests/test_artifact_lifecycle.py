import app as app_module
import workspace_store


def _configure_workspace_dirs(tmp_path, monkeypatch):
    base_dir = tmp_path / 'workspace_data'
    monkeypatch.setattr(workspace_store, 'BASE_DIR', str(base_dir))
    monkeypatch.setattr(workspace_store, 'DATASETS_DIR', str(base_dir / 'datasets'))
    monkeypatch.setattr(workspace_store, 'DASHBOARDS_DIR', str(base_dir / 'dashboards'))
    monkeypatch.setattr(workspace_store, 'REPORTS_DIR', str(base_dir / 'reports'))
    monkeypatch.setattr(workspace_store, 'RELATIONSHIPS_DIR', str(base_dir / 'relationships'))
    monkeypatch.setattr(workspace_store, 'MEASURES_DIR', str(base_dir / 'measures'))
    monkeypatch.setattr(workspace_store, 'AUDIT_DIR', str(base_dir / 'audit'))
    workspace_store.ensure_workspace_dirs()


def _set_session(client, username, dataset_id, filepath, token='lifecycle-token'):
    with client.session_transaction() as session_state:
        session_state['_csrf_token'] = token
        session_state['user'] = username
        session_state['current_dataset_id'] = dataset_id
        session_state['current_filepath'] = filepath
    return token


def test_dataset_lifecycle_can_be_promoted(tmp_path, monkeypatch):
    _configure_workspace_dirs(tmp_path, monkeypatch)
    users_file = tmp_path / 'users.json'
    monkeypatch.setattr(app_module, 'USERS_FILE', str(users_file))
    app_module._save_users({'owner': {'email': 'owner@example.com', 'password_hash': 'x'}})
    app_module.app.config['TESTING'] = True

    source = tmp_path / 'sales.csv'
    source.write_text('region,revenue\nNorth,100\n', encoding='utf-8')

    dataset_record = workspace_store.create_dataset_record(
        'owner',
        source_name='sales.csv',
        stored_path=str(source),
        source_type='upload',
        row_count=1,
        column_count=2,
        metadata={
            'display_name': 'sales.csv',
            'columns': ['region', 'revenue'],
            'lineage_steps': [],
            'pipeline_steps': [],
            'lifecycle': {'certification': 'draft', 'stage': 'dev', 'steward': 'owner', 'history': []},
        },
    )

    with app_module.app.test_client() as client:
        token = _set_session(client, 'owner', dataset_record['id'], str(source))
        response = client.post(f"/artifacts/dataset/{dataset_record['id']}/lifecycle", json={
            '_csrf_token': token,
            'certification': 'certified',
            'stage': 'prod',
            'steward': 'owner',
            'notes': 'Approved for production dashboards.',
        })
        payload = response.get_json()

    assert response.status_code == 200
    assert payload['artifact']['metadata']['lifecycle']['certification'] == 'certified'
    assert payload['artifact']['metadata']['lifecycle']['stage'] == 'prod'
    assert payload['artifact']['metadata']['lifecycle']['history'][0]['notes'] == 'Approved for production dashboards.'


def test_saved_artifacts_inherit_and_update_lifecycle(tmp_path, monkeypatch):
    _configure_workspace_dirs(tmp_path, monkeypatch)
    users_file = tmp_path / 'users.json'
    monkeypatch.setattr(app_module, 'USERS_FILE', str(users_file))
    app_module._save_users({'owner': {'email': 'owner@example.com', 'password_hash': 'x'}})
    app_module.app.config['TESTING'] = True

    source = tmp_path / 'sales.csv'
    source.write_text('region,revenue\nNorth,100\nSouth,200\n', encoding='utf-8')

    dataset_record = workspace_store.create_dataset_record(
        'owner',
        source_name='sales.csv',
        stored_path=str(source),
        source_type='upload',
        row_count=2,
        column_count=2,
        metadata={
            'display_name': 'sales.csv',
            'columns': ['region', 'revenue'],
            'lineage_steps': [],
            'pipeline_steps': [],
            'lifecycle': {'certification': 'review', 'stage': 'test', 'steward': 'owner', 'history': []},
        },
    )

    with app_module.app.test_client() as client:
        token = _set_session(client, 'owner', dataset_record['id'], str(source))
        report_response = client.post('/reports/save', json={
            '_csrf_token': token,
            'name': 'Lifecycle report',
        })
        dashboard_response = client.post('/dashboards/save', json={
            '_csrf_token': token,
            'name': 'Lifecycle dashboard',
            'dashboard_viz': [{'id': 1, 'type': 'bar', 'columns': ['region']}],
            'dashboard_state': {},
        })
        report_id = report_response.get_json()['report']['id']
        dashboard_id = dashboard_response.get_json()['dashboard']['id']

        dashboard_lifecycle_response = client.post(f'/artifacts/dashboard/{dashboard_id}/lifecycle', json={
            '_csrf_token': token,
            'certification': 'certified',
            'stage': 'prod',
            'steward': 'owner',
            'notes': 'Published to exec workspace.',
        })
        report_library_response = client.get('/report_library')
        dashboard_library_response = client.get('/dashboard_library')

    assert report_response.status_code == 200
    assert dashboard_response.status_code == 200
    assert dashboard_lifecycle_response.status_code == 200
    assert report_library_response.get_json()['reports'][0]['lifecycle']['stage'] == 'test'
    assert dashboard_library_response.get_json()['dashboards'][0]['lifecycle']['stage'] == 'prod'
