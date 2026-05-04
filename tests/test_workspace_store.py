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
