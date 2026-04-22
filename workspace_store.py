import json
import os
import re
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional


BASE_DIR = os.path.join(os.path.dirname(__file__), 'workspace_data')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
DASHBOARDS_DIR = os.path.join(BASE_DIR, 'dashboards')
RELATIONSHIPS_DIR = os.path.join(BASE_DIR, 'relationships')


def ensure_workspace_dirs() -> None:
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(DASHBOARDS_DIR, exist_ok=True)
    os.makedirs(RELATIONSHIPS_DIR, exist_ok=True)


def _safe_user_segment(username: str) -> str:
    normalized = re.sub(r'[^a-zA-Z0-9_-]+', '_', username.strip())
    return normalized or 'default_user'


def _dataset_path(username: str, dataset_id: str) -> str:
    user_dir = os.path.join(DATASETS_DIR, _safe_user_segment(username))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f'{dataset_id}.json')


def _list_user_dataset_files(username: str) -> List[str]:
    user_dir = os.path.join(DATASETS_DIR, _safe_user_segment(username))
    if not os.path.exists(user_dir):
        return []
    return [
        os.path.join(user_dir, filename)
        for filename in os.listdir(user_dir)
        if filename.endswith('.json')
    ]


def _dashboard_path(username: str, dashboard_id: str) -> str:
    user_dir = os.path.join(DASHBOARDS_DIR, _safe_user_segment(username))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f'{dashboard_id}.json')


def _list_user_dashboard_files(username: str) -> List[str]:
    user_dir = os.path.join(DASHBOARDS_DIR, _safe_user_segment(username))
    if not os.path.exists(user_dir):
        return []
    return [
        os.path.join(user_dir, filename)
        for filename in os.listdir(user_dir)
        if filename.endswith('.json')
    ]


def _relationship_path(username: str, relationship_id: str) -> str:
    user_dir = os.path.join(RELATIONSHIPS_DIR, _safe_user_segment(username))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f'{relationship_id}.json')


def _list_user_relationship_files(username: str) -> List[str]:
    user_dir = os.path.join(RELATIONSHIPS_DIR, _safe_user_segment(username))
    if not os.path.exists(user_dir):
        return []
    return [
        os.path.join(user_dir, filename)
        for filename in os.listdir(user_dir)
        if filename.endswith('.json')
    ]


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def create_dataset_record(
    username: str,
    source_name: str,
    stored_path: str,
    source_type: str,
    row_count: int,
    column_count: int,
    parent_dataset_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_workspace_dirs()
    dataset_id = f"ds_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}"
    now = datetime.utcnow().isoformat() + 'Z'

    record = {
        'id': dataset_id,
        'source_name': source_name,
        'stored_path': stored_path,
        'source_type': source_type,
        'row_count': int(row_count),
        'column_count': int(column_count),
        'parent_dataset_id': parent_dataset_id,
        'created_at': now,
        'updated_at': now,
        'metadata': metadata or {},
    }

    with open(_dataset_path(username, dataset_id), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)

    return record


def update_dataset_record(username: str, dataset_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    record = get_dataset_record(username, dataset_id)
    if not record:
        return None

    record.update(updates)
    record['updated_at'] = datetime.utcnow().isoformat() + 'Z'

    with open(_dataset_path(username, dataset_id), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)

    return record


def get_dataset_record(username: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    ensure_workspace_dirs()
    return _read_json(_dataset_path(username, dataset_id))


def list_dataset_records(username: str) -> List[Dict[str, Any]]:
    ensure_workspace_dirs()
    records = []
    for path in _list_user_dataset_files(username):
        record = _read_json(path)
        if record:
            records.append(record)

    records.sort(key=lambda item: item.get('updated_at', ''), reverse=True)
    return records


def create_dashboard_record(
    username: str,
    name: str,
    dataset_id: Optional[str],
    dashboard_viz: List[Dict[str, Any]],
    dashboard_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_workspace_dirs()
    dashboard_id = f"db_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}"
    now = datetime.utcnow().isoformat() + 'Z'
    record = {
        'id': dashboard_id,
        'name': name,
        'dataset_id': dataset_id,
        'created_at': now,
        'updated_at': now,
        'dashboard_viz': dashboard_viz,
        'dashboard_state': dashboard_state or {},
    }
    with open(_dashboard_path(username, dashboard_id), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    return record


def get_dashboard_record(username: str, dashboard_id: str) -> Optional[Dict[str, Any]]:
    ensure_workspace_dirs()
    return _read_json(_dashboard_path(username, dashboard_id))


def list_dashboard_records(username: str, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    ensure_workspace_dirs()
    records = []
    for path in _list_user_dashboard_files(username):
        record = _read_json(path)
        if not record:
            continue
        if dataset_id and record.get('dataset_id') != dataset_id:
            continue
        records.append(record)

    records.sort(key=lambda item: item.get('updated_at', ''), reverse=True)
    return records


def create_relationship_record(
    username: str,
    left_dataset_id: str,
    left_column: str,
    right_dataset_id: str,
    right_column: str,
    join_type: str,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    ensure_workspace_dirs()
    relationship_id = f"rel_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}"
    now = datetime.utcnow().isoformat() + 'Z'
    record = {
        'id': relationship_id,
        'left_dataset_id': left_dataset_id,
        'left_column': left_column,
        'right_dataset_id': right_dataset_id,
        'right_column': right_column,
        'join_type': join_type,
        'confidence': confidence,
        'created_at': now,
        'updated_at': now,
    }
    with open(_relationship_path(username, relationship_id), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    return record


def list_relationship_records(username: str) -> List[Dict[str, Any]]:
    ensure_workspace_dirs()
    records = []
    for path in _list_user_relationship_files(username):
        record = _read_json(path)
        if record:
            records.append(record)

    records.sort(key=lambda item: item.get('updated_at', ''), reverse=True)
    return records
