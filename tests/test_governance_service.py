import app as app_module
import workspace_store
from governance_service import build_governance_summary, infer_sensitivity_label


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


def _set_session(client, dataset_id, filepath, token='test-token'):
    with client.session_transaction() as session_state:
        session_state['_csrf_token'] = token
        session_state['user'] = 'analyst'
        session_state['current_dataset_id'] = dataset_id
        session_state['current_filepath'] = filepath


def test_infer_sensitivity_label_detects_confidential_fields():
    summary = {
        'semantic_profiles': [
            {'name': 'customer_email'},
            {'name': 'revenue'},
        ]
    }

    result = infer_sensitivity_label(summary)

    assert result['label'] == 'Confidential'


def test_governance_summary_route_returns_activity_and_label(tmp_path, monkeypatch):
    _configure_workspace_dirs(tmp_path, monkeypatch)
    app_module.app.config['TESTING'] = True

    source = tmp_path / 'customers.csv'
    source.write_text('customer_email,revenue\nalice@example.com,120\nbob@example.com,90\n', encoding='utf-8')

    dataset_record = workspace_store.create_dataset_record(
        'analyst',
        source_name='customers.csv',
        stored_path=str(source),
        source_type='upload',
        row_count=2,
        column_count=2,
        metadata={
            'display_name': 'customers.csv',
            'columns': ['customer_email', 'revenue'],
            'lineage_steps': [],
            'pipeline_steps': [],
        },
    )
    workspace_store.log_audit_event(
        'analyst',
        action='dataset_uploaded',
        dataset_id=dataset_record['id'],
        artifact_id=dataset_record['id'],
        details={'display_name': 'customers.csv'},
    )

    with app_module.app.test_client() as client:
        _set_session(client, dataset_record['id'], str(source))
        response = client.get('/governance_summary')
        payload = response.get_json()

    assert response.status_code == 200
    assert payload['governance']['sensitivity']['label'] == 'Confidential'
    assert payload['governance']['activity'][0]['action'] == 'dataset_uploaded'
    assert payload['governance']['pipeline']['can_rebuild'] is True


def test_build_governance_summary_counts_downstream_assets():
    dataset_record = {
        'id': 'ds_1',
        'source_type': 'upload',
        'metadata': {'pipeline_steps': [], 'lineage_steps': [], 'shared_with': [{'user': 'viewer', 'role': 'viewer'}], 'row_policies': [{'user': 'viewer', 'column': 'region', 'allowed_values': ['North']}]},
    }
    analysis_summary = {
        'semantic_profiles': [{'name': 'region'}],
        'quality_alerts': [{'severity': 'success'}],
    }

    summary = build_governance_summary(
        dataset_record,
        analysis_summary,
        audit_events=[{'action': 'dataset_uploaded'}],
        dashboards=[{'id': 'db_1'}, {'id': 'db_2'}],
        measures=[{'id': 'm_1'}],
        reports=[{'id': 'r_1'}, {'id': 'r_2'}, {'id': 'r_3'}],
        queries=[{'id': 'q_1'}],
        refresh_jobs=[{'id': 'job_1'}, {'id': 'job_2'}],
    )

    assert summary['downstream_assets'] == {'dashboards': 2, 'measures': 1, 'reports': 3, 'queries': 1, 'refresh_jobs': 2}
    assert summary['collaboration'] == {'shared_users': 1, 'row_policies': 1}


def test_governance_summary_route_counts_shared_workspace_assets(tmp_path, monkeypatch):
    _configure_workspace_dirs(tmp_path, monkeypatch)
    users_file = tmp_path / 'users.json'
    monkeypatch.setattr(app_module, 'USERS_FILE', str(users_file))
    app_module._save_users({
        'owner': {'email': 'owner@example.com', 'password_hash': 'x'},
        'viewer': {'email': 'viewer@example.com', 'password_hash': 'x'},
    })
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
            'shared_with': [{'user': 'viewer', 'role': 'viewer'}],
            'row_policies': [],
        },
    )
    workspace_store.create_dashboard_record('owner', 'Sales dashboard', dataset_record['id'], [{'id': 1}], {})
    workspace_store.create_report_record('owner', 'Sales report', dataset_record['id'], {'dataset_name': 'sales.csv', 'sections': []})
    workspace_store.create_query_record('owner', 'Sales query', dataset_record['id'], 'SELECT * FROM dataset')
    workspace_store.create_measure_record('owner', dataset_record['id'], 'Revenue', {'type': 'ratio'}, {'value': 100})
    workspace_store.create_refresh_job_record('owner', dataset_record['id'], cadence_minutes=60, mode='full')
    workspace_store.log_audit_event(
        'owner',
        action='dashboard_saved',
        dataset_id=dataset_record['id'],
        artifact_type='dashboard',
        artifact_id='db_test',
        details={'name': 'Sales dashboard'},
    )

    with app_module.app.test_client() as client:
        with client.session_transaction() as session_state:
            session_state['_csrf_token'] = 'viewer-token'
            session_state['user'] = 'viewer'
            session_state['current_dataset_id'] = dataset_record['id']
            session_state['current_filepath'] = str(source)
        response = client.get('/governance_summary')
        payload = response.get_json()

    assert response.status_code == 200
    downstream = payload['governance']['downstream_assets']
    assert downstream['dashboards'] == 1
    assert downstream['reports'] == 1
    assert downstream['queries'] == 1
    assert downstream['measures'] == 1
    assert downstream['refresh_jobs'] == 1
    assert payload['governance']['activity'][0]['action'] == 'dashboard_saved'
