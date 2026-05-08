import json

import workspace_store


def test_atomic_write_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / 'dataset.json'
    target.write_text('{"status": "original"}', encoding='utf-8')

    def failing_dump(payload, handle, indent=2):
        handle.write('{"status": ')
        raise TypeError('serialization failed')

    monkeypatch.setattr(workspace_store.json, 'dump', failing_dump)

    try:
        workspace_store._write_json_atomic(str(target), {'status': 'new'})
    except TypeError:
        pass

    assert json.loads(target.read_text(encoding='utf-8')) == {'status': 'original'}
    assert [path.name for path in tmp_path.iterdir()] == ['dataset.json']


def test_report_records_round_trip(tmp_path, monkeypatch):
    base_dir = tmp_path / 'workspace_data'
    monkeypatch.setattr(workspace_store, 'BASE_DIR', str(base_dir))
    monkeypatch.setattr(workspace_store, 'REPORTS_DIR', str(base_dir / 'reports'))
    workspace_store.ensure_workspace_dirs()

    record = workspace_store.create_report_record(
        'analyst',
        name='Weekly summary',
        dataset_id='ds_123',
        report_payload={'dataset_name': 'Sales', 'sections': [{'title': 'Overview', 'bullets': ['Rows: 10']}]},
    )

    fetched = workspace_store.get_report_record('analyst', record['id'])
    listed = workspace_store.list_report_records('analyst', dataset_id='ds_123')

    assert fetched['name'] == 'Weekly summary'
    assert fetched['report']['dataset_name'] == 'Sales'
    assert listed[0]['id'] == record['id']
    assert (base_dir / 'workspace.db').exists()


def test_legacy_json_records_migrate_into_sqlite(tmp_path, monkeypatch):
    base_dir = tmp_path / 'workspace_data'
    reports_dir = base_dir / 'reports' / 'analyst'
    reports_dir.mkdir(parents=True, exist_ok=True)
    legacy_report = reports_dir / 'rpt_legacy.json'
    legacy_report.write_text(
        json.dumps(
            {
                'id': 'rpt_legacy',
                'name': 'Legacy report',
                'dataset_id': 'ds_legacy',
                'created_at': '2026-01-01T00:00:00Z',
                'updated_at': '2026-01-02T00:00:00Z',
                'report': {'dataset_name': 'Orders', 'sections': []},
            }
        ),
        encoding='utf-8',
    )

    monkeypatch.setattr(workspace_store, 'BASE_DIR', str(base_dir))
    monkeypatch.setattr(workspace_store, 'DATASETS_DIR', str(base_dir / 'datasets'))
    monkeypatch.setattr(workspace_store, 'DASHBOARDS_DIR', str(base_dir / 'dashboards'))
    monkeypatch.setattr(workspace_store, 'REPORTS_DIR', str(base_dir / 'reports'))
    monkeypatch.setattr(workspace_store, 'RELATIONSHIPS_DIR', str(base_dir / 'relationships'))
    monkeypatch.setattr(workspace_store, 'MEASURES_DIR', str(base_dir / 'measures'))
    monkeypatch.setattr(workspace_store, 'AUDIT_DIR', str(base_dir / 'audit'))

    workspace_store.ensure_workspace_dirs()

    listed = workspace_store.list_report_records('analyst', dataset_id='ds_legacy')
    assert listed[0]['id'] == 'rpt_legacy'
    assert listed[0]['owner'] == 'analyst'
