import json

import app as app_module


def test_atomic_user_save_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    users_file = tmp_path / 'users.json'
    users_file.write_text('{"existing": {"email": "a@example.com"}}', encoding='utf-8')
    monkeypatch.setattr(app_module, 'USERS_FILE', str(users_file))

    def failing_dump(payload, handle, indent=2):
        handle.write('{"broken": ')
        raise TypeError('serialization failed')

    monkeypatch.setattr(app_module.json, 'dump', failing_dump)

    try:
        app_module._save_users({'new_user': {'email': 'b@example.com'}})
    except TypeError:
        pass

    assert json.loads(users_file.read_text(encoding='utf-8')) == {
        'existing': {'email': 'a@example.com'}
    }
    assert [path.name for path in tmp_path.iterdir()] == ['users.json']
